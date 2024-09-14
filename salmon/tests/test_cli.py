"""Tests for the command line interface."""
import pytest
from typer.testing import CliRunner

from salmon.cli import app

runner = CliRunner()


@pytest.mark.parametrize(
    "command_name",
    ["finetune", "adapt", "publish", "pull", "list", "version", "embed"],
)
def test_app_help(command_name):
    """Check that the commands can generate help messages."""
    result = runner.invoke(app, [command_name, "--help"])
    assert result.exit_code == 0
