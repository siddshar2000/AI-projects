from crewai.tools import tool
from ..utils import mcp_tool
from langchain.tools import StructuredTool

@tool('list_repo_files')
def list_repo_files(owner: str, repo: str, path = '.') -> list:
    """List files and folders at a given path in a GitHub repository."""
    # TODO: Change this
    path = "./ConferenceFindingCrewAI/"
    print(f"Repo Structure Lister: Get files at {path} for {owner}/{repo}")
    tool_command = [
        # TODO: Change this
        "tools",
        "get_file_contents",
        # "get_commit",
        # "list_branches",
        "--owner", owner,
        "--repo", repo,
        "--path", path,
        # "--issue_number", "1"
        # "--sha", "d579d4467974e69858c9b2eb93f4077dac6a3430"
        # "--help"
        # f"tools get_file_contents --help"
        ]

    tool_result = mcp_tool(
    #     [
    #     "tools",
    #     " ".join(tool_command)
    # ]
        tool_command
    )

    if tool_result is None or tool_result is not list:
        return []
    
    return tool_result

# TODO: Remove
# get_repo_files = StructuredTool.from_function(func=list_repo_files,
#                                         name='ListGitHubFilesAndFolders',
#                                         description="Tools to list GitHub files and folder for provided github path")