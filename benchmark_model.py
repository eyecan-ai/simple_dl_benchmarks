import torch
import time
import rich
import numpy as np
from rich.progress import Progress

shapes = [256, 512, 1024, 2048]
repeats = 10
device = "cuda"
model = torch.jit.load("exported_model.eye", map_location=device)

with Progress() as progress:
    benchmark = progress.add_task(
        "[red] The monkey is thinking...", total=len(shapes) * repeats
    )
    times_map = {}
    for s in shapes:
        times_map[s] = []
        for r in range(repeats):
            x = torch.rand(1, 3, s, s).to(device)
            [model(x) for _ in range(2)]
            t1 = time.perf_counter()
            out = model(x)
            t2 = time.perf_counter()
            times_map[s].append(t2 - t1)
            progress.update(benchmark, advance=1)
        times_map[s] = np.array(times_map[s]).mean()

for shape, time in times_map.items():
    hz = 1 / time
    rich.print(
        f"[red]Resolution ({shape})[/red]: [green]{time}[/green]  [blue][Hz: {hz:.2f}][/blue]"
    )
