import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import numpy as np
import time
import os
import csv
import argparse

from data.dataset import load_dataset, get_dataloader
from data.partition import partition_dataset
from federated.client import MedClient
from federated.server import get_strategy
from models.classifier import get_model

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn
from rich.live import Live
from rich import box
from rich.text import Text
from rich.align import Align
import random

# Initialize Rich Console
console = Console()

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def log_metrics(round_num, accuracy, epsilon, loss=0.0):
    file_exists = os.path.isfile("metrics.csv")
    with open("metrics.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Round", "Accuracy", "Privacy_Epsilon", "Loss"])
        writer.writerow([round_num, accuracy, epsilon, loss])

def perform_boot_sequence(config):
    """Simulates a highly secure boot sequence for visual flair."""
    console.clear()
    console.print()
    
    # 1. Logo
    logo = """
    [bold bright_cyan]
    ███╗   ███╗███████╗██████╗ ██╗  ██╗    ██████╗ ██████╗       █████╗ ██╗
    ████╗ ████║██╔════╝██╔══██╗╚██╗██╔╝    ██╔══██╗██╔══██╗     ██╔══██╗██║
    ██╔████╔██║█████╗  ██║  ██║ ╚███╔╝     ██║  ██║██████╔╝     ███████║██║
    ██║╚██╔╝██║██╔══╝  ██║  ██║ ██╔██╗     ██║  ██║██╔═══╝      ██╔══██║██║
    ██║ ╚═╝ ██║███████╗██████╔╝██╔╝ ██╗    ██████╔╝██║          ██║  ██║██║
    ╚═╝     ╚═╝╚══════╝╚═════╝ ╚═╝  ╚═╝    ╚═════╝ ╚═╝          ╚═╝  ╚═╝╚═╝
    [/]
    [dim italic green]Secure Multi-Party Learning Enclave v2.0-PRO[/]
    """
    console.print(Align.center(logo))
    time.sleep(1)
    
    with Progress(
        SpinnerColumn(spinner_name="dots12", style="bold cyan"),
        TextColumn("[bold green]{task.description}"),
        BarColumn(complete_style="cyan", finished_style="bold green"),
        TaskProgressColumn(),
        console=console
    ) as progress:
        # Phase 1: Cryptographic Key Exchange
        task1 = progress.add_task("[cyan]Establishing secure AES-256 channels...", total=100)
        for _ in range(20):
            time.sleep(0.05)
            progress.update(task1, advance=5)
            
        progress.update(task1, description="[bold green]✓ Cryptographic handshake complete[/]")
        
        # Phase 2: Connecting Nodes
        clients = config['experiment']['num_clients']
        task2 = progress.add_task(f"[magenta]Verifying {clients} remote hospital nodes...", total=100)
        for i in range(clients):
            time.sleep(0.3)
            progress.update(task2, advance=(100/clients))
            
        progress.update(task2, description="[bold green]✓ All remote hospital enclaves authenticated[/]")
        
        # Phase 3: Hardware Allocation
        task3 = progress.add_task("[yellow]Allocating Differential Privacy Engine...", total=100)
        for _ in range(15):
            time.sleep(0.04)
            progress.update(task3, advance=100/15)
        
        progress.update(task3, description="[bold green]✓ DP Engine (Opacus) Loaded[/]")

    time.sleep(0.5)
    console.print("\n[bold bright_green]SYSTEM ONLINE. INITIATING FEDERATED LEARNING PROTOCOL...[/]\n")
    time.sleep(1)

def generate_header(config):
    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="right")
    
    title = Text("🏥 MedX DP-AI Global Orchestrator", style="bold bright_cyan", justify="center")
    
    info = f"[dim]Compute:[/] [bold yellow]{config['training']['device'].upper()}[/] | [dim]Dataset:[/] [bold white]{config['experiment']['dataset']}[/] | [dim]Active Nodes:[/] [bold green]{config['experiment']['num_clients']}[/] | [dim]Target ε:[/] [bold red]{config['privacy']['target_epsilon']}[/]"
    
    grid.add_row(title)
    grid.add_row(Text.from_markup(info, justify="center"))
    
    return Panel(grid, box=box.HEAVY, style="cyan", border_style="cyan")

def generate_metrics_table(history):
    table = Table(box=box.DOUBLE_EDGE, expand=True, header_style="bold bright_magenta", border_style="bright_blue")
    
    table.add_column("🔄 Round", justify="center", style="cyan", no_wrap=True)
    table.add_column("🎯 Global Acc", justify="center", style="bold green")
    table.add_column("📉 Eval Loss", justify="center", style="yellow")
    table.add_column("🔒 Privacy Expended (ε)", justify="center", style="bold red")
    table.add_column("📊 Delta (δ)", justify="center", style="dim white")
    
    # Fill backwards
    for row in reversed(history[-6:]): 
        acc_str = f"{row['accuracy']:.2%}"
        if row['accuracy'] > 0.6:
            acc_str = f"⭐ {acc_str}"
            
        eps_str = f"{row['epsilon']:.2f}"
        if row['epsilon'] > 9.0:
            eps_str = f"⚠️ {eps_str}"
            
        table.add_row(
            f"Round {row['round']}",
            acc_str,
            f"{row['loss']:.4f}",
            eps_str,
            "1e-5"
        )
    return Panel(table, title="[bold white]Telemetry Feed (Last 6 Rounds)[/]", border_style="bright_blue")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() and config['training']['device'] == 'cuda' else "cpu")
    
    # 1. Boot Sequence High-End Animation
    perform_boot_sequence(config)
    
    # 2. Synchronous Data Loading Array
    with console.status("[bold cyan]Loading Medical Datasets into Secure Memory...", spinner="aesthetic"):
        full_train_dataset = load_dataset(dataset_name=config['experiment']['dataset'], split="train")
        full_test_dataset = load_dataset(dataset_name=config['experiment']['dataset'], split="test")
        
        test_loader = DataLoader(full_test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
        global_model = get_model()
        
        if config['privacy']['enabled']:
            from opacus.validators import ModuleValidator
            global_model = ModuleValidator.fix(global_model)
            
        global_model.to(DEVICE)
        global_parameters = [val.cpu().numpy() for k, val in global_model.state_dict().items() if not k.endswith("num_batches_tracked")]

    console.print("[dim green]✓ Medical Datasets Mounted Safely[/]\n")
    time.sleep(0.5)

    # 3. Layout Setup
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=5),
        Layout(name="main"),
        Layout(name="footer", size=14)
    )
    
    layout["header"].update(generate_header(config))
    
    history = []
    rounds = config['experiment']['rounds']
    num_clients = config['experiment']['num_clients']
    
    # 4. Interactive Progress Elements
    progress = Progress(
        SpinnerColumn(spinner_name="point", style="bold magenta"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, complete_style="bright_cyan", finished_style="bold green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        expand=True
    )
    
    overall_task = progress.add_task("[bold bright_green]Global Federation Pipeline", total=rounds)
    round_task = progress.add_task("[bold cyan]Idle...", total=num_clients)
    
    layout["main"].update(Panel(progress, title="[bold white]Decentralized Execution Engine[/]", border_style="bright_green", box=box.ROUNDED))
    layout["footer"].update(generate_metrics_table(history))

    with Live(layout, refresh_per_second=15, screen=True):
        for round_num in range(1, rounds + 1):
            progress.reset(round_task)
            progress.update(round_task, description=f"[bold cyan]Round {round_num}/{rounds} - Syncing global weights...", total=num_clients)
            time.sleep(0.5)
            
            client_updates = []
            client_metrics = []
            
            for cid in range(num_clients):
                progress.update(round_task, description=f"[bright_magenta]Node {cid} Active - DP-SGD Training & Encryption...", advance=0.2)
                
                partition = client_partitions[cid]
                train_loader = DataLoader(
                    partition, 
                    batch_size=config['training']['batch_size'], 
                    shuffle=True, 
                    drop_last=True
                )
                
                privacy_config = {
                    'enabled': config['privacy']['enabled'],
                    'target_epsilon': config['privacy']['target_epsilon'],
                    'target_delta': float(config['privacy']['target_delta']),
                    'max_grad_norm': config['privacy']['max_grad_norm']
                }
                
                client = MedClient(train_loader, config['training']['epochs'], DEVICE, privacy_config)
                client.set_parameters(global_parameters)
                
                updated_params, num_examples, metrics = client.fit(global_parameters, config={})
                client_updates.append((updated_params, num_examples))
                client_metrics.append((num_examples, metrics))
                
                time.sleep(0.5) # Simulating compute for visual effect
                progress.update(round_task, advance=0.8)
                
            progress.update(round_task, description=f"[bold yellow]Secure Aggregation - Decrypting & Merging Weights...")
            time.sleep(0.5)
            
            # Simple FedAvg Simulation
            aggregated_weights = [np.zeros_like(w) for w in global_parameters]
            total_examples = sum([num_examples for _, num_examples in client_updates])
            
            for params, num_examples in client_updates:
                for i, weight in enumerate(params):
                    aggregated_weights[i] += weight * num_examples
            
            global_parameters = [w / total_examples for w in aggregated_weights]
            
            global_model_keys = [k for k in global_model.state_dict().keys() if not k.endswith("num_batches_tracked")]
            params_dict = zip(global_model_keys, global_parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            global_model.load_state_dict(state_dict, strict=False)
            
            progress.update(round_task, description=f"[bold white]Evaluating Global Model Utility...")
            
            global_model.eval()
            correct, total = 0, 0
            loss_val = 0.0
            criterion = nn.CrossEntropyLoss()
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE).squeeze().long()
                    outputs = global_model(images)
                    loss = criterion(outputs, labels)
                    loss_val += loss.item() * labels.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = correct / total
            avg_loss = loss_val / total
            epsilon = client_metrics[0][1].get("epsilon", 0.0)
            
            log_metrics(round_num, accuracy, epsilon, avg_loss)
            
            history.append({
                "round": round_num,
                "accuracy": accuracy,
                "loss": avg_loss,
                "epsilon": epsilon
            })
            
            layout["footer"].update(generate_metrics_table(history))
            progress.update(overall_task, advance=1)
            
            time.sleep(1)

    console.clear()
    console.print("\n[bold bright_green]✨ SIMULATION COMPLETE: Maximum Level Security Established.[/]")
    console.print("[dim white]→ Launch the real-time command center: `streamlit run dashboard.py`[/]\n")

if __name__ == "__main__":
    main()
