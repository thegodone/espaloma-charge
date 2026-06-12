"""Neural network components of espaloma charge."""

import torch
import types
import sys


class MoleculeGraph:
    """Small tensor graph used by the Torch-only inference path."""

    def __init__(self, ndata, edges, batch_index=None, batch_num_nodes=None):
        self.ndata = ndata
        self.edges = edges
        n_nodes = next(iter(ndata.values())).shape[0] if ndata else 0
        device = next(iter(ndata.values())).device if ndata else torch.device("cpu")
        if batch_index is None:
            batch_index = torch.zeros(n_nodes, dtype=torch.long, device=device)
        if batch_num_nodes is None:
            batch_num_nodes = torch.tensor([n_nodes], dtype=torch.long, device=device)
        self.batch_index = batch_index
        self.batch_num_nodes = batch_num_nodes

    @property
    def batch_size(self):
        return int(self.batch_num_nodes.numel())

    @property
    def device(self):
        return self.batch_index.device

    def number_of_nodes(self):
        return next(iter(self.ndata.values())).shape[0] if self.ndata else 0

    def number_of_edges(self):
        return self.edges.shape[1]

    def to(self, device):
        self.ndata = {key: value.to(device) for key, value in self.ndata.items()}
        self.edges = self.edges.to(device)
        self.batch_index = self.batch_index.to(device)
        self.batch_num_nodes = self.batch_num_nodes.to(device)
        return self

    def mean_neighbors(self, x):
        if self.edges.numel() == 0:
            return torch.zeros_like(x)

        src, dst = self.edges
        out = torch.zeros_like(x)
        out.index_add_(0, dst, x[src])
        degree = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
        degree.index_add_(0, dst, torch.ones(dst.shape[0], 1, dtype=x.dtype, device=x.device))
        return out / degree.clamp_min(1.0)

    def sum_nodes(self, key):
        values = self.ndata[key]
        out = torch.zeros(self.batch_size, values.shape[-1], dtype=values.dtype, device=values.device)
        out.index_add_(0, self.batch_index, values)
        return out

    def broadcast_nodes(self, values):
        return values[self.batch_index]


def batch_graphs(graphs):
    if len(graphs) == 0:
        raise ValueError("cannot batch an empty graph list")

    device = graphs[0].device
    ndata = {
        key: torch.cat([graph.ndata[key].to(device) for graph in graphs], dim=0)
        for key in graphs[0].ndata
    }

    edge_blocks = []
    batch_indices = []
    batch_num_nodes = []
    offset = 0
    for batch_id, graph in enumerate(graphs):
        n_nodes = next(iter(graph.ndata.values())).shape[0]
        batch_num_nodes.append(n_nodes)
        batch_indices.append(torch.full((n_nodes,), batch_id, dtype=torch.long, device=device))
        if graph.edges.numel() > 0:
            edge_blocks.append(graph.edges.to(device) + offset)
        offset += n_nodes

    if edge_blocks:
        edges = torch.cat(edge_blocks, dim=1)
    else:
        edges = torch.empty((2, 0), dtype=torch.long, device=device)

    return MoleculeGraph(
        ndata=ndata,
        edges=edges,
        batch_index=torch.cat(batch_indices),
        batch_num_nodes=torch.tensor(batch_num_nodes, dtype=torch.long, device=device),
    )


def unbatch_graphs(graph):
    graphs = []
    node_offset = 0
    for n_nodes in graph.batch_num_nodes.tolist():
        node_slice = slice(node_offset, node_offset + n_nodes)
        ndata = {key: value[node_slice].clone() for key, value in graph.ndata.items()}

        edge_mask = (
            (graph.edges[0] >= node_offset)
            & (graph.edges[0] < node_offset + n_nodes)
            & (graph.edges[1] >= node_offset)
            & (graph.edges[1] < node_offset + n_nodes)
        )
        edges = graph.edges[:, edge_mask] - node_offset
        graphs.append(MoleculeGraph(ndata, edges))
        node_offset += n_nodes
    return graphs


class TorchSAGEConv(torch.nn.Module):
    """Torch implementation of DGL SAGEConv(mean) compatible with old checkpoints."""

    def __init__(self, in_feats=None, out_feats=None, aggregator_type="mean", bias=True, feat_drop=0.0, **kwargs):
        super().__init__()
        if aggregator_type != "mean":
            raise ValueError("TorchSAGEConv currently supports only mean aggregation")
        self._aggre_type = aggregator_type
        if in_feats is not None and out_feats is not None:
            self.fc_self = torch.nn.Linear(in_feats, out_feats, bias=False)
            self.fc_neigh = torch.nn.Linear(in_feats, out_feats, bias=False)
            self.bias = torch.nn.Parameter(torch.zeros(out_feats)) if bias else None
            self.feat_drop = torch.nn.Dropout(feat_drop)
        self.norm = kwargs.get("norm")
        self.activation = kwargs.get("activation")

    def forward(self, g, feat, **kwargs):
        h_self = self.feat_drop(feat) if hasattr(self, "feat_drop") else feat
        h_neigh = g.mean_neighbors(h_self)
        rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        bias = getattr(self, "bias", None)
        if bias is not None:
            rst = rst + bias
        activation = getattr(self, "activation", None)
        if activation is not None:
            rst = activation(rst)
        norm = getattr(self, "norm", None)
        if norm is not None:
            rst = norm(rst)
        return rst


def install_legacy_dgl_pickle_shim():
    """Allow old DGL-pickled model.pt files to load without importing DGL."""

    module_names = [
        "dgl",
        "dgl.nn",
        "dgl.nn.pytorch",
        "dgl.nn.pytorch.conv",
        "dgl.nn.pytorch.conv.sageconv",
    ]
    modules = {name: sys.modules.get(name) or types.ModuleType(name) for name in module_names}
    modules["dgl.nn.pytorch.conv.sageconv"].SAGEConv = TorchSAGEConv
    modules["dgl.nn.pytorch.conv"].sageconv = modules["dgl.nn.pytorch.conv.sageconv"]
    modules["dgl.nn.pytorch"].conv = modules["dgl.nn.pytorch.conv"]
    modules["dgl.nn"].pytorch = modules["dgl.nn.pytorch"]
    modules["dgl"].nn = modules["dgl.nn"]
    for name, module in modules.items():
        sys.modules[name] = module

class _Sequential(torch.nn.Module):
    """Sequentially staggered neural networks."""

    def __init__(
        self,
        layer,
        config,
        in_features,
        model_kwargs={},
    ):
        super(_Sequential, self).__init__()

        self.exes = []

        # init dim
        dim = in_features

        # parse the config
        for idx, exe in enumerate(config):

            try:
                exe = float(exe)

                if exe >= 1:
                    exe = int(exe)
            except BaseException:
                pass

            # int -> feedfoward
            if isinstance(exe, int):
                setattr(self, "d" + str(idx), layer(dim, exe, **model_kwargs))

                dim = exe
                self.exes.append("d" + str(idx))

            # str -> activation
            elif isinstance(exe, str):
                if exe == "bn":
                    setattr(self, "a" + str(idx), torch.nn.BatchNorm1d(dim))

                else:
                    activation = getattr(torch.nn.functional, exe)
                    setattr(self, "a" + str(idx), activation)

                self.exes.append("a" + str(idx))

            # float -> dropout
            elif isinstance(exe, float):
                dropout = torch.nn.Dropout(exe)
                setattr(self, "o" + str(idx), dropout)

                self.exes.append("o" + str(idx))

    def forward(self, g, x, **kwargs):
        for exe in self.exes:
            if exe.startswith("d"):
                if g is not None:
                    x = getattr(self, exe)(g, x)
                else:
                    x = getattr(self, exe)(x)
            else:
                x = getattr(self, exe)(x)

        return x


class Sequential(torch.nn.Module):
    """Sequential neural network with input layers.

    Parameters
    ----------
    layer : torch.nn.Module
        Graph convolution layer class. The default runtime uses ``TorchSAGEConv``.

    config : List
        A sequence of numbers (for units) and strings (for activation functions)
        denoting the configuration of the sequential model.

    feature_units : int(default=117)
        The number of input channels.

    Methods
    -------
    forward(g, x)
        Forward pass.
    """

    def __init__(
        self,
        layer,
        config,
        feature_units=116,
        input_units=128,
        model_kwargs={},
    ):
        super(Sequential, self).__init__()

        # initial featurization
        self.f_in = torch.nn.Sequential(
            torch.nn.Linear(feature_units, input_units), torch.nn.Tanh()
        )

        self._sequential = _Sequential(
            layer, config, in_features=input_units, model_kwargs=model_kwargs
        )

    def _forward(self, g, x):
        """Forward pass with graph and features."""
        for exe in self.exes:
            if exe.startswith("d"):
                x = getattr(self, exe)(g, x)
            else:
                x = getattr(self, exe)(x)

        return x

    def forward(self, g, x=None, **kwargs):
        """Forward pass.

        Parameters
        ----------
        g : `MoleculeGraph`,
            input graph

        Returns
        -------
        g : `MoleculeGraph`
            output graph
        """
        if x is None:
            # get node attributes
            x = g.ndata["h0"]
            x = self.f_in(x)

        # message passing on homo graph
        x = self._sequential(g, x)

        # put attribute back in the graph
        g.ndata["h"] = x

        return g

def _total_charge_tensor(total_charge, batch_size, device, dtype):
    if total_charge is None:
        total_charge = 0.0

    if isinstance(total_charge, torch.Tensor):
        total_charge = total_charge.to(device=device, dtype=dtype)
    else:
        total_charge = torch.as_tensor(total_charge, device=device, dtype=dtype)

    if total_charge.ndim == 0:
        total_charge = total_charge.reshape(1).expand(batch_size)
    else:
        total_charge = total_charge.reshape(-1)
        if total_charge.numel() == 1 and batch_size != 1:
            total_charge = total_charge.expand(batch_size)

    if total_charge.numel() != batch_size:
        raise ValueError(
            f"total_charge must be a scalar or have one value per molecule; "
            f"got {total_charge.numel()} values for batch_size={batch_size}"
        )

    return total_charge.reshape(batch_size, 1)


def qeq_charges(e, s, total_charge, batch_index):
    s_inv = s ** -1
    e_s_inv = e * s_inv

    batch_size = total_charge.shape[0]
    sum_s_inv = torch.zeros(batch_size, s.shape[-1], dtype=s.dtype, device=s.device)
    sum_e_s_inv = torch.zeros(batch_size, e.shape[-1], dtype=e.dtype, device=e.device)
    sum_s_inv.index_add_(0, batch_index, s_inv)
    sum_e_s_inv.index_add_(0, batch_index, e_s_inv)

    return -e * s_inv + s_inv * torch.div(
        total_charge[batch_index] + sum_e_s_inv[batch_index],
        sum_s_inv[batch_index],
    )


def get_charges(node):
    """ Solve the function to get the absolute charges of atoms in a
    molecule from parameters.
    Parameters
    ----------
    e : tf.Tensor, dtype = tf.float32,
        electronegativity.
    s : tf.Tensor, dtype = tf.float32,
        hardness.
    Q : tf.Tensor, dtype = tf.float32, shape=(),
        total charge of a molecule.
    We use Lagrange multipliers to analytically give the solution.
    $$
    U({\bf q})
    &= \sum_{i=1}^N \left[ e_i q_i +  \frac{1}{2}  s_i q_i^2\right]
        - \lambda \, \left( \sum_{j=1}^N q_j - Q \right) \\
    &= \sum_{i=1}^N \left[
        (e_i - \lambda) q_i +  \frac{1}{2}  s_i q_i^2 \right
        ] + Q
    $$
    This gives us:
    $$
    q_i^*
    &= - e_i s_i^{-1}
    + \lambda s_i^{-1} \\
    &= - e_i s_i^{-1}
    + s_i^{-1} \frac{
        Q +
         \sum\limits_{i=1}^N e_i \, s_i^{-1}
        }{\sum\limits_{j=1}^N s_j^{-1}}
    $$
    """
    e = node.data["e"]
    s = node.data["s"]
    sum_e_s_inv = node.data["sum_e_s_inv"]
    sum_s_inv = node.data["sum_s_inv"]
    sum_q = node.data["sum_q"]

    return {
        "q": -e * s**-1
        + (s**-1) * torch.div(sum_q + sum_e_s_inv, sum_s_inv)
    }

class ChargeReadout(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc_params = torch.nn.Linear(in_features, 2)

    def forward(self, g, **kwargs):
        h = self.fc_params(g.ndata["h"])
        e, s = h.split(1, -1)
        g.ndata["e"], g.ndata["s"] = e, s
        return g

class ChargeEquilibrium(torch.nn.Module):
    """Charge equilibrium within batches of molecules."""

    def __init__(self):
        super(ChargeEquilibrium, self).__init__()

    def forward(self, g, total_charge=None):
        """apply charge equilibrium to all molecules in batch"""
        if total_charge is None and "q_ref" in g.ndata:
            total_charge = g.sum_nodes("q_ref")
        else:
            total_charge = _total_charge_tensor(
                total_charge,
                batch_size=g.batch_size,
                device=g.device,
                dtype=g.ndata["s"].dtype,
            )

        g.ndata["q"] = qeq_charges(
            g.ndata["e"],
            g.ndata["s"],
            total_charge,
            g.batch_index,
        )

        return g
