
import numpy as np
import os

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch_geometric.nn import conv
from torch_geometric.utils import from_scipy_sparse_matrix

from scipy import sparse
import math


# Get the paths of the repository
file_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(file_path))


class TMFModel(pl.LightningModule):
    def __init__(self, args):
        super(TMFModel, self).__init__()
        self.args = args

        self.save_hyperparameters()
        self.linear_embedding = LinearEmbedding(3,self.args)
        self.pos_encoder= PositionalEncoding1D(self.args.latent_size)
        self.encoder_transformer = EncoderTransformer(self.args)

        self.agent_gnn = AgentGnn(self.args)
        self.decoder_residual = DecoderResidual(self.args)

        self.reg_loss = nn.SmoothL1Loss(reduction="none")
        self.is_frozen = False

    @staticmethod
    def init_args(parent_parser):
        parser_dataset = parent_parser.add_argument_group("dataset")
        parser_dataset.add_argument(
            "--train_split", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "train"))
        parser_dataset.add_argument(
            "--val_split", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "val"))
        parser_dataset.add_argument(
            "--test_split", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "test"))
        parser_dataset.add_argument(
            "--train_split_pre", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "train_pre.pkl"))
        parser_dataset.add_argument(
            "--val_split_pre", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "val_pre.pkl"))
        parser_dataset.add_argument(
            "--test_split_pre", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "test_pre.pkl"))
        parser_dataset.add_argument(
            "--reduce_dataset_size", type=int, default=0)
        parser_dataset.add_argument(
            "--use_preprocessed", type=bool, default=False)
        parser_dataset.add_argument(
            "--align_image_with_target_x", type=bool, default=True)

        parser_training = parent_parser.add_argument_group("training")
        parser_training.add_argument("--num_epochs", type=int, default=72)
        parser_training.add_argument(
            "--lr_values", type=list, default=[1e-3, 1e-4, 1e-3 , 1e-4])
        parser_training.add_argument(
            "--lr_step_epochs", type=list, default=[5, 20, 68])
        parser_training.add_argument("--wd", type=float, default=0.01)
        parser_training.add_argument("--batch_size", type=int, default=32)
        parser_training.add_argument("--val_batch_size", type=int, default=32)
        parser_training.add_argument("--workers", type=int, default=0)
        parser_training.add_argument("--val_workers", type=int, default=0)
        parser_training.add_argument("--gpus", type=int, default=1)

        parser_model = parent_parser.add_argument_group("model")
        parser_model.add_argument("--latent_size", type=int, default=128)
        parser_model.add_argument("--num_preds", type=int, default=60)
        parser_model.add_argument("--mod_steps", type=list, default=[1, 5])
        parser_model.add_argument("--mod_freeze_epoch", type=int, default=20)

        return parent_parser

    def forward(self, batch):
        # Set batch norm to eval mode in order to prevent updates on the running means,
        # if the weights are frozen
        if self.is_frozen:
            for module in self.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d):
                    module.eval()

        displ, centers = batch["displ"], batch["centers"]
        rotation, origin = batch["rotation"], batch["origin"]

        # Extract the number of agents in each sample of the current batch
        agents_per_sample = [x.shape[0] for x in displ]
       
        # Convert the list of tensors to tensors
        displ_cat = torch.cat(displ, dim=0)
    
        centers_cat = torch.cat(centers, dim=0)
        
        linear_output = self.linear_embedding(displ_cat)
        pos_encodeing = self.pos_encoder(linear_output)
        pos_encodeing = pos_encodeing + linear_output

        out_transformer = self.encoder_transformer(pos_encodeing, agents_per_sample)

        out_agent_gnn = self.agent_gnn(out_transformer, centers_cat, agents_per_sample)


        out_agent_gnn= torch.stack([x[0] for x in out_agent_gnn])
        
        out_linear = self.decoder_residual(out_agent_gnn, self.is_frozen)

        out = out_linear.view(len(displ), 1, -1, self.args.num_preds, 2)

        # Iterate over each batch and transform predictions into the global coordinate frame
        for i in range(len(out)):
            out[i] = torch.matmul(out[i], rotation[i]) + origin[i].view(
                1, 1, 1, -1
            )
        return out

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        self.decoder_residual.unfreeze_layers()

        self.is_frozen = True

    def prediction_loss(self, preds, gts):
        # Stack all the predicted trajectories of the target agent
        num_mods = preds.shape[2]
        # [0] is required to remove the unneeded dimensions
        preds = torch.cat([x[0] for x in preds], 0)

        # Stack all the true trajectories of the target agent
        # Keep in mind, that there are multiple trajectories in each sample, but only the first one ([0]) corresponds
        # to the target agent
        gt_target = torch.cat([torch.unsqueeze(x[0], 0) for x in gts], 0)
        gt_target = torch.repeat_interleave(gt_target, num_mods, dim=0)

        loss_single = self.reg_loss(preds, gt_target)
        loss_single = torch.sum(torch.sum(loss_single, dim=2), dim=1)

        loss_single = torch.split(loss_single, num_mods)

        # Tuple to tensor
        loss_single = torch.stack(list(loss_single), dim=0)

        min_loss_index = torch.argmin(loss_single, dim=1)

        min_loss_combined = [x[min_loss_index[i]]
                             for i, x in enumerate(loss_single)]

        loss_out = torch.sum(torch.stack(min_loss_combined))

        return loss_out

    def configure_optimizers(self):

        # self.args.mod_freeze_epoch = 0  # addeddddd 
        if self.current_epoch == self.args.mod_freeze_epoch:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), weight_decay=self.args.wd)
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), weight_decay=self.args.wd)
        return optimizer

    def on_train_epoch_start(self):
 
        # self.args.mod_freeze_epoch = 0 # addeddd
        # Trigger weight freeze and optimizer reinit on mod_freeze_epoch
        if self.current_epoch == self.args.mod_freeze_epoch:
            self.freeze()
            self.trainer.accelerator_backend.setup_optimizers(self.trainer)


        # Set learning rate according to current epoch
        for single_param in self.optimizers().param_groups:
            single_param["lr"] = self.get_lr(self.current_epoch)


    def training_step(self, train_batch, batch_idx):
        out = self.forward(train_batch)
        loss = self.prediction_loss(out, train_batch["gt"])
        self.log("loss_train", loss / len(out))
        return loss

    def get_lr(self, epoch):
        lr_index = 0
        for lr_epoch in self.args.lr_step_epochs:
            if epoch < lr_epoch:
                break
            lr_index += 1
        
        return self.args.lr_values[lr_index]

    def validation_step(self, val_batch, batch_idx):
        out = self.forward(val_batch)
        loss = self.prediction_loss(out, val_batch["gt"])
        self.log("loss_val", loss / len(out))

        # Extract target agent only
        pred = [x[0].detach().cpu().numpy() for x in out]
        gt = [x[0].detach().cpu().numpy() for x in val_batch["gt"]]
        return pred, gt

    def validation_epoch_end(self, validation_outputs):
        # Extract predictions
        pred = [out[0] for out in validation_outputs]
        pred = np.concatenate(pred, 0)
        gt = [out[1] for out in validation_outputs]
        gt = np.concatenate(gt, 0)
        ade1, fde1, ade, fde = self.calc_prediction_metrics(pred, gt)
        self.log("ade1_val", ade1, prog_bar=True)
        self.log("fde1_val", fde1, prog_bar=True)
        self.log("ade_val", ade, prog_bar=True)
        self.log("fde_val", fde, prog_bar=True)

    def calc_prediction_metrics(self, preds, gts):
        # Calculate prediction error for each mode
        # Output has shape (batch_size, n_modes, n_timesteps)
        error_per_t = np.linalg.norm(preds - np.expand_dims(gts, axis=1), axis=-1)

        # Calculate the error for the first mode (at index 0)
        fde_1 = np.average(error_per_t[:, 0, -1])
        ade_1 = np.average(error_per_t[:, 0, :])

        # Calculate the error for all modes
        # Best mode is always the one with the lowest final displacement
        lowest_final_error_indices = np.argmin(error_per_t[:, :, -1], axis=1)
        error_per_t = error_per_t[np.arange(
            preds.shape[0]), lowest_final_error_indices]
        fde = np.average(error_per_t[:, -1])
        ade = np.average(error_per_t[:, :])

        return ade_1, fde_1, ade, fde



class LinearEmbedding(nn.Module):
    def __init__(self,input_size,args):
        super(LinearEmbedding, self).__init__()
        self.args = args
        self.input_size = input_size
        self.output_size = args.latent_size

        self.encoder_input_layer = nn.Linear(
                in_features=self.input_size, 
                out_features=self.output_size 
                    )
    def forward(self,linear_input):

        linear_out = F.relu(self.encoder_input_layer(linear_input))

        return linear_out 


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)

        return self.cached_penc

class EncoderTransformer(nn.Module):
    def __init__(self, args):
        super(EncoderTransformer, self).__init__()
        self.args = args

        self.d_model = self.args.latent_size # embedding dimension
        self.nhead = 128
        self.d_hid = 1 ## dimension of the feedforward network model in nn.TransformerEncoder
        self.num_layers = 1
        self.dropout = 0.0

        self.encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.nhead, self.d_hid , self.dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

    def forward(self, transformer_in, agents_per_sample):

        transformer_out = F.relu(self.transformer_encoder(transformer_in))
        return transformer_out[:,-1,:]

class AgentGnn(nn.Module):
    def __init__(self, args):
        super(AgentGnn, self).__init__()
        self.args = args
        self.latent_size = args.latent_size

        self.gcn1 = conv.CGConv(self.latent_size, dim=2, batch_norm=True)
        self.gcn2 = conv.CGConv(self.latent_size, dim=2, batch_norm=True)

    def forward(self, gnn_in, centers, agents_per_sample):
        # gnn_in is a batch and has the shape (batch_size, number_of_agents, latent_size)

        x, edge_index = gnn_in, self.build_fully_connected_edge_idx(
            agents_per_sample).to(gnn_in.device)
        edge_attr = self.build_edge_attr(edge_index, centers).to(gnn_in.device)

        x = F.relu(self.gcn1(x, edge_index, edge_attr))
        gnn_out = F.relu(self.gcn2(x, edge_index, edge_attr))

        edge_index_out1 = []
        for i in agents_per_sample:
            edge_index_out1.append(gnn_out[0:i,:])
            gnn_out = gnn_out[i:,:]

        return edge_index_out1

    def build_fully_connected_edge_idx(self, agents_per_sample):
        edge_index = []

        # In the for loop one subgraph is built (no self edges!)
        # The subgraph gets offsetted and the full graph over all samples in the batch
        # gets appended with the offsetted subgrah
        offset = 0
        for i in range(len(agents_per_sample)):

            num_nodes = agents_per_sample[i]

            adj_matrix = torch.ones((num_nodes, num_nodes))
            adj_matrix = adj_matrix.fill_diagonal_(0)

            sparse_matrix = sparse.csr_matrix(adj_matrix.numpy())
            edge_index_subgraph, _ = from_scipy_sparse_matrix(sparse_matrix)

            # Offset the list
            edge_index_subgraph = torch.Tensor(
                np.asarray(edge_index_subgraph) + offset)
            offset += agents_per_sample[i]

            edge_index.append(edge_index_subgraph)

        # Concat the single subgraphs into one
        edge_index = torch.LongTensor(np.column_stack(edge_index))
        

        return edge_index
    def build_edge_attr(self, edge_index, data):
        edge_attr = torch.zeros((edge_index.shape[-1], 2), dtype=torch.float)

        rows, cols = edge_index
        # goal - origin
        edge_attr = data[cols] - data[rows]

        return edge_attr

class DecoderResidual(nn.Module):
    def __init__(self, args):
        super(DecoderResidual, self).__init__()

        self.args = args

        output = []
        for i in range(sum(args.mod_steps)):
            output.append(PredictionNet(args))

        self.output = nn.ModuleList(output)

    def forward(self, decoder_in, is_frozen):
        sample_wise_out = []

        if self.training is False:
            for out_subnet in self.output:
                sample_wise_out.append(out_subnet(decoder_in))
        elif is_frozen:
            for i in range(self.args.mod_steps[0], sum(self.args.mod_steps)):
                sample_wise_out.append(self.output[i](decoder_in))
        else:
            sample_wise_out.append(self.output[0](decoder_in))
            

        decoder_out = torch.stack(sample_wise_out)
        decoder_out = torch.swapaxes(decoder_out, 0, 1)

        return decoder_out

    def unfreeze_layers(self):
        for layer in range(self.args.mod_steps[0], sum(self.args.mod_steps)):
            for param in self.output[layer].parameters():
                param.requires_grad = True


class PredictionNet(nn.Module):
    def __init__(self, args):
        super(PredictionNet, self).__init__()

        self.args = args

        self.latent_size = args.latent_size

        self.weight1 = nn.Linear(self.latent_size, self.latent_size)
        self.norm1 = nn.GroupNorm(1, self.latent_size)

        self.weight2 = nn.Linear(self.latent_size, self.latent_size)
        self.norm2 = nn.GroupNorm(1, self.latent_size)

        self.output_fc = nn.Linear(self.latent_size, args.num_preds * 2)

    def forward(self, prednet_in):
        # Residual layer
        x = self.weight1(prednet_in)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.weight2(x)
        x = self.norm2(x)

        x += prednet_in

        x = F.relu(x)

        # Last layer has no activation function
        prednet_out = self.output_fc(x)

        return prednet_out
