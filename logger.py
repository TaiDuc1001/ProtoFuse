import logging
from typing import Any, Dict, List, Optional

from rich.theme import Theme
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich.rule import Rule
from rich.align import Align
from rich.console import Console
from rich.logging import RichHandler
from rich.box import ROUNDED, DOUBLE, HEAVY, MINIMAL    
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn


custom_theme = Theme({
    "info": "bold cyan",
    "warning": "bold yellow",
    "error": "bold red",
    "critical": "bold white on red",
    "debug": "dim cyan",
    "success": "bold green",
    "highlight": "bold magenta",
    "muted": "dim white",
    "repr.number": "bold green",
    "repr.str": "bold magenta",
    "metric.name": "cyan",
    "metric.value": "bold white",
    "metric.good": "bold green",
    "metric.bad": "bold red",
    "header": "bold blue on white",
})

console = Console(theme=custom_theme, force_terminal=True)

ICONS = {
    "debug": "🔍",
    "info": "ℹ️ ",
    "warning": "⚠️ ",
    "error": "❌",
    "critical": "🔥",
    "success": "✅",
    "start": "🚀",
    "end": "🏁",
    "config": "⚙️ ",
    "model": "🧠",
    "data": "📊",
    "train": "🏋️",
    "eval": "📈",
    "save": "💾",
    "load": "📂",
    "time": "⏱️ ",
    "epoch": "🔄",
    "best": "🏆",
    "checkpoint": "📍",
}


class RichLogger:
    def __init__(self, name: str = "apt"):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers = []
        self._console = console
        self._show_icons = True

    def _setup_handler(self, level: int) -> None:
        self._logger.handlers = []
        handler = RichHandler(
            console=self._console,
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True,
            log_time_format="[%H:%M:%S]",
            keywords=["accuracy", "loss", "epoch", "lr", "best", "saved"],
        )
        handler.setLevel(level)
        self._logger.addHandler(handler)
        self._logger.setLevel(level)

    def _icon(self, key: str) -> str:
        if not self._show_icons:
            return ""
        return ICONS.get(key, "") + " "

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        self._logger.exception(msg, *args, **kwargs)

    def success(self, msg: str) -> None:
        self._console.print(f"[bold green]{self._icon('success')}[/bold green][green]{msg}[/green]")

    def header(self, title: str, subtitle: str = "", style: str = "bold blue") -> None:
        header_text = Text()
        header_text.append(f" {title} ", style=style)
        if subtitle:
            header_text.append(f"\n{subtitle}", style="dim")
        panel = Panel(
            Align.center(header_text),
            box=DOUBLE,
            border_style=style,
            padding=(1, 2),
        )
        self._console.print(panel)

    def section(self, title: str, icon_key: str = "info") -> None:
        icon = self._icon(icon_key)
        self._console.print()
        self._console.print(Rule(f"[bold cyan]{icon}{title}[/bold cyan]", style="cyan", align="left"))

    def subsection(self, title: str) -> None:
        self._console.print(f"  [dim]──[/dim] [bold white]{title}[/bold white]")

    def divider(self, style: str = "dim") -> None:
        self._console.print(Rule(style=style))

    def metrics(self, metrics_dict: Dict[str, Any], title: str = "Metrics", highlight_best: Optional[str] = None) -> None:
        table = Table(
            title=f"[bold]{title}[/bold]",
            box=ROUNDED,
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            padding=(0, 1),
        )
        table.add_column("Metric", style="cyan", justify="right")
        table.add_column("Value", style="bold white", justify="left")

        for key, value in metrics_dict.items():
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)

            if highlight_best and key == highlight_best:
                formatted_value = f"[bold green]★ {formatted_value}[/bold green]"

            table.add_row(key, formatted_value)

        self._console.print(table)

    def config_table(self, config: Dict[str, Any], title: str = "Configuration") -> None:
        table = Table(
            title=f"[bold]{self._icon('config')}{title}[/bold]",
            box=MINIMAL,
            show_header=True,
            header_style="bold magenta",
            border_style="dim",
        )
        table.add_column("Parameter", style="magenta", justify="right")
        table.add_column("Value", style="white", justify="left")

        for key, value in config.items():
            if isinstance(value, dict):
                value_str = "{...}"
            elif isinstance(value, list) and len(value) > 5:
                value_str = f"[{len(value)} items]"
            else:
                value_str = str(value)
            table.add_row(key, value_str)

        self._console.print(table)

    def epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        train_acc: float,
        val_loss: Optional[float] = None,
        val_acc: Optional[float] = None,
        lr: Optional[float] = None,
        is_best: bool = False,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        progress_bar = "█" * int((epoch / total_epochs) * 20) + "░" * (20 - int((epoch / total_epochs) * 20))

        header = f"[bold cyan]{self._icon('epoch')}Epoch {epoch}/{total_epochs}[/bold cyan]"
        if is_best:
            header += f" [bold yellow]{self._icon('best')}NEW BEST[/bold yellow]"

        content = Text()
        content.append(f"\n  Progress: [{progress_bar}] {epoch/total_epochs*100:.1f}%\n\n", style="dim")
        content.append(f"  📉 Train Loss: ", style="dim")
        content.append(f"{train_loss:.4f}\n", style="bold white")
        content.append(f"  🎯 Train Acc:  ", style="dim")
        content.append(f"{train_acc:.2%}\n", style="bold green" if train_acc > 0.8 else "bold white")

        if val_loss is not None:
            content.append(f"  📊 Val Loss:   ", style="dim")
            content.append(f"{val_loss:.4f}\n", style="bold white")
        if val_acc is not None:
            content.append(f"  ✨ Val Acc:    ", style="dim")
            acc_style = "bold green" if val_acc > 0.8 else ("bold yellow" if val_acc > 0.5 else "bold red")
            content.append(f"{val_acc:.2%}\n", style=acc_style)
        if lr is not None:
            content.append(f"  📐 LR:         ", style="dim")
            content.append(f"{lr:.2e}\n", style="bold cyan")

        if extra_metrics:
            content.append("\n", style="dim")
            for key, value in extra_metrics.items():
                content.append(f"  • {key}: ", style="dim")
                content.append(f"{value:.4f}\n", style="bold white")

        border_style = "green" if is_best else "blue"
        panel = Panel(
            content,
            title=header,
            box=HEAVY if is_best else ROUNDED,
            border_style=border_style,
            padding=(0, 1),
        )
        self._console.print(panel)

    def model_summary(self, model_name: str, num_params: int, trainable_params: int, extra_info: Optional[Dict[str, str]] = None) -> None:
        tree = Tree(f"[bold blue]{self._icon('model')}Model: {model_name}[/bold blue]")
        tree.add(f"[cyan]Total Parameters:[/cyan] [bold]{num_params:,}[/bold]")
        tree.add(f"[cyan]Trainable:[/cyan] [bold green]{trainable_params:,}[/bold green] ({trainable_params/num_params*100:.1f}%)")
        tree.add(f"[cyan]Frozen:[/cyan] [bold yellow]{num_params - trainable_params:,}[/bold yellow]")

        if extra_info:
            info_branch = tree.add("[cyan]Details[/cyan]")
            for key, value in extra_info.items():
                info_branch.add(f"[dim]{key}:[/dim] {value}")

        self._console.print(Panel(tree, box=ROUNDED, border_style="blue", padding=(0, 1)))

    def training_start(self, experiment_name: str, config_summary: Optional[Dict[str, Any]] = None) -> None:
        self.header(
            f"{self._icon('start')}Starting Training",
            subtitle=experiment_name,
            style="bold green",
        )
        if config_summary:
            self.config_table(config_summary, "Run Configuration")

    def training_end(self, best_acc: float, total_time: str, message: str = "") -> None:
        content = Text()
        content.append(f"\n  {self._icon('best')} Best Accuracy: ", style="dim")
        content.append(f"{best_acc:.2%}\n", style="bold green")
        content.append(f"  {self._icon('time')} Total Time: ", style="dim")
        content.append(f"{total_time}\n", style="bold cyan")
        if message:
            content.append(f"\n  {message}\n", style="italic")

        panel = Panel(
            content,
            title=f"[bold green]{self._icon('end')}Training Complete[/bold green]",
            box=DOUBLE,
            border_style="green",
            padding=(0, 2),
        )
        self._console.print(panel)

    def checkpoint_saved(self, path: str, is_best: bool = False) -> None:
        icon = self._icon('best') if is_best else self._icon('checkpoint')
        style = "bold green" if is_best else "cyan"
        msg = "Best model" if is_best else "Checkpoint"
        self._console.print(f"  [{style}]{icon}{msg} saved:[/{style}] [dim]{path}[/dim]")

    def progress_context(self, description: str, total: Optional[int] = None):
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self._console,
            transient=True,
        )

    def step(self, step_num: int, total_steps: int, loss: float, acc: Optional[float] = None, extra: str = "") -> None:
        pct = step_num / total_steps * 100
        bar_len = 20
        filled = int(bar_len * step_num / total_steps)
        bar = "━" * filled + "╺" + "─" * (bar_len - filled - 1)

        msg = f"  [dim]Step {step_num:>5}/{total_steps}[/dim] [{bar}] [cyan]{pct:5.1f}%[/cyan] | loss: [bold]{loss:.4f}[/bold]"
        if acc is not None:
            msg += f" | acc: [bold green]{acc:.2%}[/bold green]"
        if extra:
            msg += f" | {extra}"

        self._console.print(msg, end="\r")

    def step_done(self) -> None:
        self._console.print()

    def comparison_table(self, rows: List[Dict[str, Any]], columns: List[str], title: str = "Results Comparison") -> None:
        table = Table(
            title=f"[bold]{title}[/bold]",
            box=ROUNDED,
            show_header=True,
            header_style="bold blue",
        )

        for col in columns:
            table.add_column(col, justify="center")

        for row in rows:
            values = []
            for col in columns:
                val = row.get(col, "-")
                if isinstance(val, float):
                    val = f"{val:.4f}"
                values.append(str(val))
            table.add_row(*values)

        self._console.print(table)


logger = RichLogger()


def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logger._setup_handler(level)
