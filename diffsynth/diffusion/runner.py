import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from diffsynth.core import OffloadTrainingManager


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    enable_model_cpu_offload: bool = False,
    enable_optimizer_cpu_offload: bool = False,
    cpu_offload_split_threshold: int = None,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    if enable_model_cpu_offload:
        optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
        model.pipe.device = accelerator.device
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
    else:
        model.to(device=accelerator.device)
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    initialize_deepspeed_gradient_checkpointing(accelerator)
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    if enable_model_cpu_offload:
        dataloader = accelerator.prepare(dataloader)
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
        model.pipe.device = accelerator.device
    else:
        model.to(device=accelerator.device)
        model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()

def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")

def collate_cached_training_inputs(batch):
    return collate_training_values(batch)

def collate_training_values(values):
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(values, dim=0) if first.ndim > 0 and first.shape[0] == 1 else torch.stack(values, dim=0)
    if isinstance(first, tuple):
        return tuple(collate_training_values([value[index] for value in values]) for index in range(len(first)))
    if isinstance(first, dict):
        return {key: collate_training_values([value[key] for value in values]) for key in first}
    return values

def launch_dmd_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    enable_model_cpu_offload: bool = False,
    enable_optimizer_cpu_offload: bool = False,
    cpu_offload_split_threshold: int = None,
    tau = 0.0,
    dmd_step = 4,
    dfake_gen_ratio = 5,
    args = None,
):
    dmd_batch_size = 1
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        if float(args.dmd_tau) != 0.0:
            raise ValueError("DMD training currently supports only --dmd_tau 0.0.")
        dmd_step = args.dmd_step
        dfake_gen_ratio = args.dmd_dfake_gen_ratio
        dmd_batch_size = getattr(args, "dmd_batch_size", dmd_batch_size)
    tau = 0.0

    generator_params = model.dmd_generator_parameters()
    fake_score_params = model.dmd_fake_score_parameters()
    if len(generator_params) == 0:
        raise ValueError("No trainable generator parameters found for DMD.")
    if len(fake_score_params) == 0:
        raise ValueError("No trainable fake-score parameters found for DMD.")

    gen_optimizer = torch.optim.AdamW(generator_params, lr=learning_rate, weight_decay=weight_decay)
    fake_optimizer = torch.optim.AdamW(fake_score_params, lr=learning_rate, weight_decay=weight_decay)
    gen_scheduler = torch.optim.lr_scheduler.ConstantLR(gen_optimizer)
    fake_scheduler = torch.optim.lr_scheduler.ConstantLR(fake_optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=dmd_batch_size, shuffle=True, collate_fn=collate_cached_training_inputs, num_workers=num_workers)

    if enable_model_cpu_offload:
        gen_optimizer, fake_optimizer, dataloader, gen_scheduler, fake_scheduler = accelerator.prepare(
            gen_optimizer, fake_optimizer, dataloader, gen_scheduler, fake_scheduler
        )
        model.pipe.device = accelerator.device
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
    else:
        model.to(device=accelerator.device)
        model, gen_optimizer, fake_optimizer, dataloader, gen_scheduler, fake_scheduler = accelerator.prepare(
            model, gen_optimizer, fake_optimizer, dataloader, gen_scheduler, fake_scheduler
        )

    initialize_deepspeed_gradient_checkpointing(accelerator)
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            for _ in range(dfake_gen_ratio):
                with accelerator.accumulate(model):
                    accelerator.unwrap_model(model).set_dmd_update_mode("fake_score")
                    # Data is actually never used
                    if dataset.load_from_cache:
                        loss_fake = model({}, inputs=data, dmd_update="fake_score", tau=tau, dmd_step=dmd_step)
                    else:
                        loss_fake = model(data, dmd_update="fake_score", tau=tau, dmd_step=dmd_step)
                    accelerator.backward(loss_fake)
                    if enable_model_cpu_offload:
                        offload_manager.after_backward()
                    print(f"**********Fake loss is: {loss_fake}***********")
                    fake_optimizer.step()
                    fake_scheduler.step()
                    fake_optimizer.zero_grad()
                    model_logger.on_step_end(accelerator, model, save_steps, loss_fake=loss_fake)

            with accelerator.accumulate(model):
                accelerator.unwrap_model(model).set_dmd_update_mode("generator")
                if dataset.load_from_cache:
                    loss_dm = model({}, inputs=data, dmd_update="generator", tau=tau, dmd_step=dmd_step)
                else:
                    loss_dm = model(data, dmd_update="generator", tau=tau, dmd_step=dmd_step)
                accelerator.backward(loss_dm)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()
                print(f"###########Generator loss is: {loss_dm}#########")
                gen_optimizer.step()
                gen_scheduler.step()
                gen_optimizer.zero_grad()
                model_logger.on_step_end(accelerator, model, save_steps, loss_dm=loss_dm)
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)
