from builtins import enumerate
import os
import argparse
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch import nn
import datetime
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_absolute_error, roc_auc_score
from tqdm import tqdm
import numpy as np
import random
from pretrain3d.data.spice import SPICE
from pretrain3d.data.process_matbench import MATBENCH
from pretrain3d.model.gnn import GNNet
from torch.optim.lr_scheduler import LambdaLR
from pretrain3d.utils.misc import WarmCosine, PreprocessBatch
from pretrain3d.utils.dist import init_distributed_mode
import json
from collections import defaultdict
from torch.utils.data import DistributedSampler
from matbench.bench import MatbenchBenchmark
from matbench.constants import CLF_KEY
from sklearn.preprocessing import StandardScaler
import wandb
from pathlib import Path


# torch.set_float32_matmul_precision('high')

def train(model, device, loader, optimizer, scheduler, args, preprocessor):
    model.train()
    loss_accum_dict = defaultdict(float)
    loss_log_dict = defaultdict(float)
    pbar = tqdm(loader, desc="Train Iteration", disable=args.disable_tqdm)
    for step, batch in enumerate(pbar):
        batch = batch.to(device)
        preprocessor.process(batch)
        optimizer.zero_grad()
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            loss_accum = 0
            for mode in args.tasks:
                pred_attrs, attr_mask_index, pos_predictions, pos_mask_idx = model(batch, mode=mode)
                if args.distributed:
                    loss, loss_dict = model.module.compute_loss(
                        pred_attrs,
                        attr_mask_index,
                        pos_predictions,
                        pos_mask_idx,
                        batch,
                        args,
                        mode=mode,
                    )
                else:
                    loss, loss_dict = model.compute_loss(
                        pred_attrs,
                        attr_mask_index,
                        pos_predictions,
                        pos_mask_idx,
                        batch,
                        args,
                        mode=mode,
                    )
                loss_accum = loss_accum + loss
                for k, v in loss_dict.items():
                    loss_accum_dict[f"{mode}_{k}"] += v
            loss_accum_dict["loss"] += loss_accum.item()
            loss_accum.backward()
            if args.grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)

            optimizer.step()
            scheduler.step()

            if step % args.log_interval == 0:
                description = f"Iteration loss: {loss_accum_dict['loss'] / (step + 1):6.4f}"
                description += f" lr: {scheduler.get_last_lr()[0]:.5e}"
                pbar.set_description(description)

    for k in loss_accum_dict.keys():
        loss_accum_dict[k] /= step + 1
    return loss_accum_dict


def evaluate(model, scaler, device, loader, args, preprocessor, model_train=False, normalize=True):
    model.eval()
    if model_train:
        model.train()
    loss_accum_dict = defaultdict(float)
    target_vals, predict_vals = [], []
    pbar = tqdm(loader, desc="Valid Iteration", disable=args.disable_tqdm)
    for step, batch in enumerate(pbar):
        batch = batch.to(device)
        target_vals.extend(batch.target.cpu().detach().numpy())
        preprocessor.process(batch)
        with torch.no_grad(): 
            loss_accum = 0
            for mode in args.tasks:
                pred_attrs, attr_mask_index, pos_predictions, pos_mask_idx = model(batch, mode=mode)
                if normalize:
                    pred_attrs = scaler.inverse_transform(pred_attrs.squeeze(-1).cpu().detach().numpy().reshape(-1,1))
                    predict_vals.extend(pred_attrs.squeeze(-1))
                else:
                    predict_vals.extend(pred_attrs.squeeze(-1).cpu().detach().numpy())
                
                if args.distributed:
                    loss, loss_dict = model.module.compute_loss(
                        pred_attrs,
                        attr_mask_index,
                        pos_predictions,
                        pos_mask_idx,
                        batch,
                        args,
                        mode=mode,
                    )
                else:
                    loss, loss_dict = model.compute_loss(
                        torch.tensor(pred_attrs).to(args.device),
                        attr_mask_index,
                        pos_predictions,
                        pos_mask_idx,
                        batch,
                        args,
                        mode=mode,
                    )

                loss_accum = loss_accum + loss
                for k, v in loss_dict.items():
                    loss_accum_dict[f"{mode}_{k}"] += v
            
            loss_accum_dict["loss"] += loss_accum.item()

    for k in loss_accum_dict.keys():
        loss_accum_dict[k] /= step + 1
    
    
    return loss_accum_dict, target_vals, predict_vals



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--global-reducer", type=str, default="sum")
    parser.add_argument("--node-reducer", type=str, default="sum")
    parser.add_argument("--graph-pooling", type=str, default="sum")
    parser.add_argument("--dropedge-rate", type=float, default=0.1)
    parser.add_argument("--dropnode-rate", type=float, default=0.1)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--latent-size", type=int, default=256)
    parser.add_argument("--mlp-hidden-size", type=int, default=1024)
    parser.add_argument("--mlp-layers", type=int, default=2)
    parser.add_argument("--use-layer-norm", action="store_true", default=False)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--results_dir", type=str, default="./results")

    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--encoder-dropout", type=float, default=0.0)
    parser.add_argument("--pooler-dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--layernorm-before", action="store_true", default=False)
    parser.add_argument("--use-bn", action="store_true", default=False)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--period", type=float, default=10)
    parser.add_argument("--enable-tb", action="store_true", default=False)
    parser.add_argument("--enable-wandb", action="store_true", default=False)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-subset", action="store_true", default=False)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--use-face", action="store_true", default=False)
    parser.add_argument("--global-attn", action="store_true", default=False)
    parser.add_argument("--node-attn", action="store_true", default=False)
    parser.add_argument("--face-attn", action="store_true", default=False)
    parser.add_argument("--grad-norm", type=float, default=None)
    parser.add_argument("--random-rotation", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--pred-pos-residual", action="store_true", default=False)
    parser.add_argument("--raw-with-pos", action="store_true", default=False)

    parser.add_argument("--eval-from", type=str, default=None)
    parser.add_argument("--pos-mask-prob", type=float, default=None)
    # parser.add_argument("--tasks", type=str, nargs="*", default=["matbench"])
    parser.add_argument("--tasks", type=str, nargs="*", default=["matbench"])
    parser.add_argument("--dataset_name", type=str, default="matbench_jdft2d")
    parser.add_argument("--restore", action="store_true", default=False)

    parser.add_argument('--energy_loss_weight', type=float, default=1.0)
    parser.add_argument('--force_loss_weight', type=float, default=1.0)
    parser.add_argument('--max-force', type=float, default=100)
    parser.add_argument('--loss_type', type=str, default='huber')
    parser.add_argument('--huber-delta', type=float, default=0.01)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--task", action="store_true", default=False)

    args = parser.parse_args()
    os.environ["WANDB_API_KEY"] = "a1645f5f73193eab34aae47a4375b5aebcd519fb"
    args.enable_wandb = "WANDB_API_KEY" in os.environ
    args.enable_wandb = False
    if args.enable_wandb:  
        now_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        wandb.init(project='matbench',
                name=f"{args.dataset_name}"+"_"+f"{now_time}")

 
    init_distributed_mode(args)
    
    assert len(args.tasks) >= 1
    assert all([task in ["mask", "mol2conf", "conf2mol", "ff", "matbench"] for task in args.tasks])
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
 
    print(args.dataset_name)

    matbench = MatbenchBenchmark(
            autoload=False,
            subset=[
                f"{args.dataset_name}"
                # "matbench_jdft2d",
                # "matbench_dielectric",
                # "matbench_phonons",
                # "matbench_perovskites",
                # "matbench_log_gvrh",
                # "matbench_log_kvrh",
                # "matbench_mp_e_form",
                # "matbench_mp_gap",
                # "matbench_mp_is_metal",
            ],
        )
    
    shared_params = dict(
                mlp_hidden_size=args.mlp_hidden_size,
                mlp_layers=args.mlp_layers,
                latent_size=args.latent_size,
                use_layer_norm=args.use_layer_norm,
                num_message_passing_steps=args.num_layers,
                global_reducer=args.global_reducer,
                node_reducer=args.node_reducer,
                face_reducer=args.node_reducer,
                dropedge_rate=args.dropedge_rate,
                dropnode_rate=args.dropnode_rate,
                use_face=args.use_face,
                dropout=args.dropout,
                graph_pooling=args.graph_pooling,
                layernorm_before=args.layernorm_before,
                encoder_dropout=args.encoder_dropout,
                pooler_dropout=args.pooler_dropout,
                use_bn=args.use_bn,
                global_attn=args.global_attn,
                node_attn=args.node_attn,
                face_attn=args.face_attn,
                mask_prob=args.mask_prob,
                pos_mask_prob=args.pos_mask_prob if args.pos_mask_prob is not None else args.mask_prob,
                pred_pos_residual=args.pred_pos_residual,
                raw_with_pos=args.raw_with_pos,
                attr_predict=True,
                num_tasks=1 # Energy only. Forece is computed in backward
            )


    for task in matbench.tasks:
        save_path = args.results_dir + f"/{task.dataset_name}"
        os.makedirs(save_path, exist_ok=True)

        task.load()
        args.classification = (task.metadata["task_type"] == "classification")
        maes, roc_aucs = [], [] 

        for fold in task.folds:
            # if fold==1:
            #     break
            intermediate_results_ = save_path + f"/fold_{fold}_targets_predictions.csv" 
            if os.path.exists(intermediate_results_):
                # If there are already intermediate results stored, skip the training.
                print('Load intermediate results')
                df = pd.read_csv(intermediate_results_)
                target_vals = df.target_vals.values
                predict_vals = df.predict_vals.values
                
            else:
                train_df = task.get_train_and_val_data(fold, as_type="df")
                test_df = task.get_test_data(fold, include_target=True, as_type="df")
                target = [ col for col in train_df.columns
                                if col not in ("id", "structure", "composition")
                                ][0]  
                
                scaler = StandardScaler()
                if args.normalize and not args.classification:
                    train_df[target] = scaler.fit_transform(train_df[target].to_numpy().reshape(-1,1))
                
                if args.classification:  # is_metal
                    train_df["is_metal"] = train_df["is_metal"].astype(int)
                    test_df["is_metal"] = test_df["is_metal"].astype(int)

                # dataset preprocessing
                # train_df =  train_df[:500]
                # test_df = test_df[:128]
                train_dataset = MATBENCH(root=f"./dataset/MATBENCH/{task.dataset_name}/fold_{fold}", 
                                            data=train_df, target=target, split="train")
                test_dataset = MATBENCH(root=f"./dataset/MATBENCH/{task.dataset_name}/fold_{fold}", 
                                        data=test_df, target=target, split="test")

                
                train_size = len(train_dataset)
                test_size = len(test_dataset)
                print(f"Dataset: {task.dataset_name}_fold_{fold} \n" 
                        "train size:", train_size, "test size:", test_size, "target:", target)
                
                if args.distributed:
                    sampler_train = DistributedSampler(train_dataset)
                else:
                    sampler_train = torch.utils.data.RandomSampler(train_dataset)

                batch_sampler_train = torch.utils.data.BatchSampler(
                    sampler_train, args.batch_size, drop_last=True
                )
                train_loader = DataLoader(
                    train_dataset, batch_sampler=batch_sampler_train, num_workers=args.num_workers
                )
                test_loader = DataLoader(
                    test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
                )

                
                model = GNNet(**shared_params).to(device)
                
                train_preprocessor = PreprocessBatch(norm2origin=True, random_rotation=args.random_rotation)
                test_preprocessor = PreprocessBatch(norm2origin=True, random_rotation=False)
                
                args.disable_tqdm = False
                if args.eval_from is not None:
                    assert os.path.exists(args.eval_from)
                    checkpoint = torch.load(args.eval_from, map_location=torch.device("cpu"))[
                        "model_state_dict"
                    ]
                    model.load_state_dict(checkpoint)

                    print("model train")
                    train_loss_dict = evaluate(
                        model, device, train_loader, args, train_preprocessor, model_train=True
                    )
                    print("train", json.dumps(train_loss_dict))
                    valid_loss_dict = evaluate(
                        model, device, test_loader, args, test_preprocessor, model_train=True
                    )
                    print("valid", json.dumps(valid_loss_dict))

                    print("model eval")
                    train_loss_dict = evaluate(model, device, train_loader, args, train_preprocessor)
                    print("train", json.dumps(train_loss_dict))
                    valid_loss_dict = evaluate(model, device, test_loader, args, test_preprocessor)
                    print("valid", json.dumps(valid_loss_dict))

                    exit(0)

                restore_fn = os.path.join(args.checkpoint_dir, "checkpoint_last.pt")
                if args.restore:
                    if os.path.exists(restore_fn):
                        print(f"Restore from {restore_fn}")
                        restore_checkpint = torch.load(restore_fn, map_location=torch.device("cpu"))
                        model.load_state_dict(restore_checkpint["model_state_dict"])
                    else:
                        args.restore = False

                model_without_ddp = model
                if args.distributed:
                    model = torch.nn.parallel.DistributedDataParallel(
                        model,
                        device_ids=[args.local_rank],
                        broadcast_buffers=False,
                        find_unused_parameters=True,
                    )
                    args.checkpoint_dir = "" if args.rank != 0 else args.checkpoint_dir
                    args.enable_tb = False if args.rank != 0 else args.enable_tb
                    args.enable_wandb = args.rank == 0 and args.enable_wandb
                    args.disable_tqdm = args.rank != 0
                
                num_params = sum(p.numel() for p in model_without_ddp.parameters())
                print(f"#Params: {num_params}")

                optimizer = optim.AdamW(
                    model_without_ddp.parameters(),
                    lr=args.lr,
                    betas=(0.9, args.beta2),
                    weight_decay=args.weight_decay,
                )
                lrscheduler = WarmCosine(tmax=len(train_loader) * args.period, warmup=int(4e3))
                scheduler = LambdaLR(optimizer, lambda x: lrscheduler.step(x))
                if args.restore:
                    optimizer.load_state_dict(restore_checkpint["optimizer_state_dict"])
                    scheduler.load_state_dict(restore_checkpint["scheduler_state_dict"])

                if args.checkpoint_dir:
                    os.makedirs(args.checkpoint_dir, exist_ok=True)

                if args.checkpoint_dir and args.enable_tb:
                    tb_writer = SummaryWriter(args.checkpoint_dir)
                
                start_epoch = restore_checkpint["epoch"] if args.restore else 0
                min_mae = restore_checkpint["min_mae"] if args.restore else float("inf")
                min_roc = restore_checkpint["min_roc"] if args.restore else float("inf")
                
                
                for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
                    if args.distributed:
                        sampler_train.set_epoch(epoch)
                    print("=====Epoch {}".format(epoch))
                    print("Training...")
                    train_loss_dict = train(model_without_ddp, device, train_loader, optimizer, scheduler, args, train_preprocessor)
                    if args.enable_wandb:
                        for k, v in train_loss_dict.items():
                            wandb.log({f"fold_{fold}_training/{k}": v })

                    if epoch % 10 == 0:
                        print("Evaluating...")
                        valid_loss_dict, target_vals, predict_vals = evaluate(model_without_ddp, scaler, device, test_loader, args, test_preprocessor, normalize=args.normalize)
                        print(
                            f"Train loss: {train_loss_dict['loss']:6.4f} Valid loss: {valid_loss_dict['loss']:6.4f}"
                        )

                        if not args.classification:
                            eval_mae = mean_absolute_error(target_vals, predict_vals)
                            if args.enable_wandb:
                                wandb.log({f"fold_{fold}_MAE": eval_mae})
                            print(f"Current MAE: {round(eval_mae, 4)} at epoch {epoch}")
                        else:
                            eval_roc = roc_auc_score(target_vals,predict_vals)
                            if args.enable_wandb:
                                wandb.log({f"fold_{fold}_ROC": eval_roc})
                            print(f"Current ROC: {round(eval_roc, 4)} at epoch {epoch}")
        
                print("Finished traning!")
                print("Testing...")
                valid_loss_dict, target_vals, predict_vals = evaluate(model_without_ddp, scaler, device, test_loader, args, test_preprocessor, normalize=args.normalize)
                vals = list(zip(target_vals, predict_vals))
                df3 = pd.DataFrame(data=vals, columns=['target_vals', 'predict_vals'])
                df3.to_csv(intermediate_results_, index=False)

            if not args.classification:
                mae = mean_absolute_error(target_vals, predict_vals)
                maes.append(mae)
                print(f"{args.dataset_name}_fold_{fold} test dataset MAE: {mae}")
            else:
                roc = roc_auc_score(target_vals,predict_vals)
                roc_aucs.append(roc)
                print(f"{args.dataset_name}_fold_{fold} test dataset ROC: {roc}")

            task.record(fold, predict_vals, params=shared_params)
                    

        
        # Result 
        if not args.classification:
            maes = np.array(maes)
            print(f"Task: {task.dataset_name} mean mae:{np.mean(maes)} std mae:{np.std(maes)}")
        else:
            roc_aucs = np.array(roc_aucs)
            print(f"Task: {task.dataset_name} mean roc_aucs:{np.mean(roc_aucs)} std roc_aucs:{np.std(roc_aucs)}")


        if args.checkpoint_dir and args.enable_tb:
            tb_writer.close()
        if args.distributed:
            torch.distributed.destroy_process_group()
        if args.enable_wandb:
            wandb.finish()
        

if __name__ == "__main__":
    main()
