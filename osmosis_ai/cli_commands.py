from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .auth import (
    LoginError,
    delete_credentials,
    delete_workspace_credentials,
    get_all_workspaces,
    get_active_workspace,
    load_credentials,
    login,
    set_active_workspace,
)
from .cli_services import (
    CLIError,
    ParsedItem,
    BaselineComparator,
    ConsoleReportRenderer,
    DatasetLoader,
    EvaluationSession,
    EvaluationSessionRequest,
    JsonReportWriter,
    RubricEvaluationEngine,
    RubricSuite,
    discover_rubric_config_path,
    load_jsonl_records,
    load_rubric_configs,
    load_rubric_suite,
    render_json_records,
    render_yaml_items,
)


class PreviewCommand:
    """Handler for `osmosis preview`."""

    def __init__(
        self,
        *,
        yaml_loader: Callable[[Path], list[ParsedItem]] = load_rubric_configs,
        json_loader: Callable[[Path], list[dict[str, Any]]] = load_jsonl_records,
    ):
        self._yaml_loader = yaml_loader
        self._json_loader = json_loader

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.set_defaults(handler=self.run)
        parser.add_argument(
            "-p",
            "--path",
            dest="path",
            required=True,
            help="Path to the YAML or JSONL file to inspect.",
        )

    def run(self, args: argparse.Namespace) -> int:
        path = Path(args.path).expanduser()
        if not path.exists():
            raise CLIError(f"Path '{path}' does not exist.")
        if path.is_dir():
            raise CLIError(f"Expected a file path but got directory '{path}'.")

        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            items = self._yaml_loader(path)
            print(f"Loaded {len(items)} rubric config(s) from {path}")
            print(render_yaml_items(items, label="Rubric config"))
        elif suffix == ".jsonl":
            records = self._json_loader(path)
            print(f"Loaded {len(records)} JSONL record(s) from {path}")
            print(render_json_records(records))
        else:
            raise CLIError(f"Unsupported file extension '{suffix}'. Expected .yaml, .yml, or .jsonl.")

        return 0


class EvalCommand:
    """Handler for `osmosis eval`."""

    def __init__(
        self,
        *,
        session: Optional[EvaluationSession] = None,
        config_locator: Callable[[Optional[str], Path], Path] = discover_rubric_config_path,
        suite_loader: Callable[[Path], RubricSuite] = load_rubric_suite,
        dataset_loader: Optional[DatasetLoader] = None,
        engine: Optional[RubricEvaluationEngine] = None,
        renderer: Optional[ConsoleReportRenderer] = None,
        report_writer: Optional[JsonReportWriter] = None,
        baseline_comparator: Optional[BaselineComparator] = None,
    ):
        self._renderer = renderer or ConsoleReportRenderer()
        if session is not None:
            self._session = session
        else:
            self._session = EvaluationSession(
                config_locator=config_locator,
                suite_loader=suite_loader,
                dataset_loader=dataset_loader,
                engine=engine,
                baseline_comparator=baseline_comparator,
                report_writer=report_writer,
                identifier_factory=self._generate_output_identifier,
            )

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.set_defaults(handler=self.run)
        parser.add_argument(
            "-r",
            "--rubric",
            dest="rubric_id",
            required=True,
            help="Rubric identifier declared in the rubric config file.",
        )
        parser.add_argument(
            "-d",
            "--data",
            dest="data_path",
            required=True,
            help="Path to the JSONL file containing evaluation records.",
        )
        parser.add_argument(
            "-n",
            "--number",
            dest="number",
            type=int,
            default=1,
            help="Run the evaluation multiple times to sample provider variance (default: 1).",
        )
        parser.add_argument(
            "-c",
            "--config",
            dest="config_path",
            help="Path to the rubric config YAML (defaults to searching near the data file).",
        )
        parser.add_argument(
            "-o",
            "--output",
            dest="output_path",
            help="Optional path to write evaluation results as JSON.",
        )
        parser.add_argument(
            "-b",
            "--baseline",
            dest="baseline_path",
            help="Optional path to a prior evaluation JSON to compare against.",
        )

    def run(self, args: argparse.Namespace) -> int:
        rubric_id_raw = getattr(args, "rubric_id", "")
        rubric_id = str(rubric_id_raw).strip()
        if not rubric_id:
            raise CLIError("Rubric identifier cannot be empty.")

        data_path = Path(args.data_path).expanduser()
        config_path_value = getattr(args, "config_path", None)
        output_path_value = getattr(args, "output_path", None)
        baseline_path_value = getattr(args, "baseline_path", None)

        number_value = getattr(args, "number", None)
        number = int(number_value) if number_value is not None else 1

        request = EvaluationSessionRequest(
            rubric_id=rubric_id,
            data_path=data_path,
            number=number,
            config_path=Path(config_path_value).expanduser() if config_path_value else None,
            output_path=Path(output_path_value).expanduser() if output_path_value else None,
            baseline_path=Path(baseline_path_value).expanduser() if baseline_path_value else None,
        )

        try:
            result = self._session.execute(request)
        except KeyboardInterrupt:
            print("Evaluation cancelled by user.")
            return 1
        self._renderer.render(result.report, result.baseline)

        if result.written_path is not None:
            print(f"Wrote evaluation results to {result.written_path}")

        return 0

    @staticmethod
    def _generate_output_identifier() -> str:
        return str(int(time.time()))


class LoginCommand:
    """Handler for `osmosis login`."""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.set_defaults(handler=self.run)
        parser.add_argument(
            "-f",
            "--force",
            dest="force",
            action="store_true",
            help="Force re-login, clearing existing credentials.",
        )
        parser.add_argument(
            "--no-browser",
            dest="no_browser",
            action="store_true",
            help="Don't open browser automatically, just print the URL.",
        )

    def run(self, args: argparse.Namespace) -> int:
        ascii_art = """
                       ___           ___           ___           ___           ___                       ___
            ___       /\\  \\         /\\  \\         /\\__\\         /\\  \\         /\\  \\          ___        /\\  \\
      __   /\\__\\     /::\\  \\       /::\\  \\       /::|  |       /::\\  \\       /::\\  \\        /\\  \\      /::\\  \\
    /\\__\\  \\/__/    /:/\\:\\  \\     /:/\\ \\  \\     /:|:|  |      /:/\\:\\  \\     /:/\\ \\  \\       \\:\\  \\    /:/\\ \\  \\
   /:/  /  /\\__\\   /:/  \\:\\  \\   _\\:\\~\\ \\  \\   /:/|:|__|__   /:/  \\:\\  \\   _\\:\\~\\ \\  \\      /::\\__\\  _\\:\\~\\ \\  \\
  /:/  /  /:/  /  /:/__/ \\:\\__\\ /\\ \\:\\ \\ \\__\\ /:/ |::::\\__\\ /:/__/ \\:\\__\\ /\\ \\:\\ \\ \\__\\  __/:/\\/__/ /\\ \\:\\ \\ \\__\\
  \\/__/  /:/  /   \\:\\  \\ /:/  / \\:\\ \\:\\ \\/__/ \\/__/~~/:/  / \\:\\  \\ /:/  / \\:\\ \\:\\ \\/__/ /\\/:/  /    \\:\\ \\:\\ \\/__/
  /\\__\\  \\/__/     \\:\\  /:/  /   \\:\\ \\:\\__\\         /:/  /   \\:\\  /:/  /   \\:\\ \\:\\__\\   \\::/__/      \\:\\ \\:\\__\\
  \\/__/             \\:\\/:/  /     \\:\\/:/  /        /:/  /     \\:\\/:/  /     \\:\\/:/  /    \\:\\__\\       \\:\\/:/  /
                     \\::/  /       \\::/  /        /:/  /       \\::/  /       \\::/  /      \\/__/        \\::/  /
                      \\/__/         \\/__/         \\/__/         \\/__/         \\/__/                     \\/__/

"""
        print(ascii_art)

        try:
            # Clear existing credentials if forcing re-login
            if args.force and load_credentials():
                delete_credentials()

            result = login(no_browser=args.no_browser)

            print(f"\n[OK] Logged in as {result.user.email}")
            if result.user.name:
                print(f"    Name: {result.user.name}")
            print(f"    Workspace: {result.organization.name} ({result.organization.role})")
            print(f"    Token expires: {result.expires_at.strftime('%Y-%m-%d')}")
            if result.revoked_previous_tokens > 0:
                token_word = "token" if result.revoked_previous_tokens == 1 else "tokens"
                print(f"    [Note] {result.revoked_previous_tokens} previous {token_word} for this device was revoked")
            print(f"    Credentials saved to ~/.config/osmosis/credentials.json")
            return 0

        except LoginError as e:
            print(f"\n[ERROR] {e}")
            return 1
        except KeyboardInterrupt:
            print("\n\nLogin cancelled.")
            return 1


class WhoamiCommand:
    """Handler for `osmosis whoami`."""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.set_defaults(handler=self.run)

    def run(self, _args: argparse.Namespace) -> int:
        workspaces = get_all_workspaces()

        if not workspaces:
            print("Not logged in. Run 'osmosis login' to authenticate.")
            return 1

        # Find active workspace for user info
        active_creds = None
        for _, creds, is_active in workspaces:
            if is_active:
                active_creds = creds
                break

        # Show user info from active workspace
        if active_creds:
            print(f"Email: {active_creds.user.email}")
            if active_creds.user.name:
                print(f"Name: {active_creds.user.name}")

        # Show all workspaces
        print(f"\nWorkspaces ({len(workspaces)}):")
        for name, creds, is_active in workspaces:
            active_marker = " *" if is_active else ""
            expired_marker = " [expired]" if creds.is_expired() else ""
            print(f"  {name} ({creds.organization.role}){active_marker}{expired_marker}")

        return 0


class LogoutCommand:
    """Handler for `osmosis logout`."""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.set_defaults(handler=self.run)
        parser.add_argument(
            "--all",
            dest="logout_all",
            action="store_true",
            help="Logout from all workspaces.",
        )
        parser.add_argument(
            "-y", "--yes",
            dest="skip_confirm",
            action="store_true",
            help="Skip confirmation prompt.",
        )

    def run(self, args: argparse.Namespace) -> int:
        workspaces = get_all_workspaces()

        if not workspaces:
            print("Not logged in.")
            return 0

        if args.logout_all:
            return self._logout_all(workspaces, args.skip_confirm)
        else:
            return self._logout_interactive(workspaces, args.skip_confirm)

    def _logout_all(
        self,
        workspaces: list[tuple[str, Any, bool]],
        skip_confirm: bool,
    ) -> int:
        """Logout from all workspaces."""
        workspace_names = [name for name, _, _ in workspaces]

        if not skip_confirm:
            print(f"This will logout from {len(workspaces)} workspace(s):")
            for name in workspace_names:
                print(f"  - {name}")
            confirm = input("\nAre you sure? [y/N]: ").strip().lower()
            if confirm not in ("y", "yes"):
                print("Cancelled.")
                return 0

        success_count = 0
        for name, _, _ in workspaces:
            if delete_workspace_credentials(name):
                success_count += 1

        print(f"Logged out from {success_count}/{len(workspaces)} workspace(s).")
        return 0

    def _logout_interactive(
        self,
        workspaces: list[tuple[str, Any, bool]],
        skip_confirm: bool,
    ) -> int:
        """Interactive workspace selection for logout."""
        if len(workspaces) == 1:
            # Only one workspace, logout directly
            name, _, _ = workspaces[0]
            if not skip_confirm:
                confirm = input(f"Logout from '{name}'? [y/N]: ").strip().lower()
                if confirm not in ("y", "yes"):
                    print("Cancelled.")
                    return 0
            if delete_workspace_credentials(name):
                print(f"Logged out from '{name}'.")
            return 0

        # Multiple workspaces, show selection
        print("Select workspace to logout from:\n")
        for i, (name, creds, is_active) in enumerate(workspaces, 1):
            active_marker = " (active)" if is_active else ""
            expired_marker = " [expired]" if creds.is_expired() else ""
            print(f"  {i}. {name}{active_marker}{expired_marker}")
        print(f"  {len(workspaces) + 1}. All workspaces")
        print(f"  0. Cancel")

        try:
            choice = input("\nEnter number: ").strip()
            choice_num = int(choice)
        except (ValueError, EOFError):
            print("Cancelled.")
            return 0

        if choice_num == 0:
            print("Cancelled.")
            return 0
        elif choice_num == len(workspaces) + 1:
            return self._logout_all(workspaces, skip_confirm)
        elif 1 <= choice_num <= len(workspaces):
            name, _, _ = workspaces[choice_num - 1]
            if not skip_confirm:
                confirm = input(f"Logout from '{name}'? [y/N]: ").strip().lower()
                if confirm not in ("y", "yes"):
                    print("Cancelled.")
                    return 0
            if delete_workspace_credentials(name):
                print(f"Logged out from '{name}'.")
            return 0
        else:
            print("Invalid selection.")
            return 1


class WorkspaceCommand:
    """Handler for `osmosis workspace`."""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        subparsers = parser.add_subparsers(dest="workspace_action", help="Workspace management commands")

        # workspace list
        list_parser = subparsers.add_parser("list", help="List all logged-in workspaces")
        list_parser.set_defaults(handler=self._run_list)

        # workspace switch
        switch_parser = subparsers.add_parser("switch", help="Switch to a different workspace")
        switch_parser.add_argument("name", help="Name of the workspace to switch to")
        switch_parser.set_defaults(handler=self._run_switch)

        # workspace current
        current_parser = subparsers.add_parser("current", help="Show the current active workspace")
        current_parser.set_defaults(handler=self._run_current)

        # Default handler when no subcommand is provided
        parser.set_defaults(handler=self._run_default)

    def _run_default(self, args: argparse.Namespace) -> int:
        """Show help when no subcommand is provided."""
        print("Usage: osmosis workspace <command>")
        print("")
        print("Commands:")
        print("  list     List all logged-in workspaces")
        print("  switch   Switch to a different workspace")
        print("  current  Show the current active workspace")
        return 0

    def _run_list(self, args: argparse.Namespace) -> int:
        """List all stored workspaces."""
        workspaces = get_all_workspaces()

        if not workspaces:
            print("No workspaces logged in. Run 'osmosis login' to log in to a workspace.")
            return 0

        print("Logged-in workspaces:")
        for name, creds, is_active in workspaces:
            active_marker = " (active)" if is_active else ""
            expired_marker = " [expired]" if creds.is_expired() else ""
            print(f"  {name} ({creds.organization.role}){active_marker}{expired_marker}")

        return 0

    def _run_switch(self, args: argparse.Namespace) -> int:
        """Switch to a different workspace."""
        workspace_name = args.name

        if set_active_workspace(workspace_name):
            print(f"Switched to workspace: {workspace_name}")
            return 0
        else:
            print(f"Workspace '{workspace_name}' not found.")
            print("Run 'osmosis workspace list' to see available workspaces.")
            return 1

    def _run_current(self, args: argparse.Namespace) -> int:
        """Show the current active workspace."""
        active = get_active_workspace()

        if not active:
            print("No active workspace. Run 'osmosis login' to log in to a workspace.")
            return 0

        workspaces = get_all_workspaces()
        for name, creds, is_active in workspaces:
            if is_active:
                expired_marker = " [expired]" if creds.is_expired() else ""
                print(f"Current workspace: {name} ({creds.organization.role}){expired_marker}")
                print(f"  User: {creds.user.email}")
                print(f"  Expires: {creds.expires_at.strftime('%Y-%m-%d')}")
                return 0

        print(f"Current workspace: {active}")
        return 0
