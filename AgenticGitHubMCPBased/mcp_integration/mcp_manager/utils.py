
import shlex
import subprocess
import json
import os
from django.conf import settings
from django.utils.encoding import smart_str

def mcp_tool(command_args: list[str]) -> dict or list or str or None: # pyright: ignore[reportInvalidTypeForm]
    """
    Executes mcpcurl with the given command arguments and returns the JSON response.
    """
    # mcpcurl_path = os.path.join(os.getcwd(), '..\\github-mcp-server\\mcpcurl')
    # mcpcurl_path = "C:\\Users\\hi_si\\source\\repos\\AI-projects\\AgenticGitHubMCPBased\\github-mcp-server\\mcpcurl"
    mcpcurl_path = "C:/Users/hi_si/source/repos/AI-projects/AgenticGitHubMCPBased/github-mcp-server/mcpcurl"
    # TODO: Change the path
    base_command = [mcpcurl_path, 
                    "--stdio-server-cmd",
                    "C:/Users/hi_si/source/repos/AI-projects/AgenticGitHubMCPBased/github-mcp-server/cmd/github-mcp-server/github-mcp-server stdio",
                    # "--stdio-server-cmd=\"C:\\Users\\hi_si\\source\\repos\\AI-projects\\AgenticGitHubMCPBased\\github-mcp-server\\cmd\\github-mcp-server\\github-mcp-server stdio\"",
                    #  "--toolsets=\"repos,issues,pull_requests,code_security\""
                     ]
    full_command = base_command + command_args
    # full_command = base_command + ["schema"]
    # full_command = base_command[:-1] + [base_command[-1] + " " + " ".join(command_args)]
    # full_command = ' '.join(full_command)
    # command = mcpcurl_path + " " + " ".join(command_args) + ' --stdio-server-cmd=\"C:\\Users\\hi_si\\source\\repos\\AI-projects\\AgenticGitHubMCPBased\\github-mcp-server\\cmd\\github-mcp-server\\github-mcp-server stdio --toolsets repos,issues,pull_requests,code_security\"'
    # full_command = shlex.split(command)
    env = {'GITHUB_PERSONAL_ACCESS_TOKEN': settings.GITHUB_PUBLIC_ACCESS_TOKEN}

    smart_cmd = [smart_str(x) for x in full_command]
    print(f"mcp_tool executing command: {smart_cmd}")  # Debug log

    try:
        process = subprocess.Popen(smart_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env=env, text=True)
        stdout, stderr = process.communicate(timeout=20)
        if stderr:
            print(f"mcpcurl stderr: {stderr}")  # Debug log

        # TODO: Remove
        print(f"mcpcurl stdout: {stdout}")  # Debug log

        if stdout:
            try:
                return json.loads(stdout)
            except json.JSONDecodeError:
                print(f"mcpcurl stdout is not valid JSON: {stdout}")
                return stdout.strip()
        else:
            return None

    except FileNotFoundError:
        print(f"Error: mcpcurl not found at {mcpcurl_path}")
        return None
    except subprocess.TimeoutExpired:
        print("Error: Timeout communicating with mcpcurl.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while running mcpcurl: {e}")
        return None



# def mcp_tool(command_args: list[str]) -> dict or list or str or None:
#     """
#     Executes mcpcurl with the given command arguments and returns the JSON response.
#     """
#     mcpcurl_path = os.path.join(os.getcwd(), 'mcpcurl')  # Assuming mcpcurl is in the project root
#     base_command = [mcpcurl_path, '--stdio-server-cmd',
#                     f'/usr/local/bin/github-mcp-server --toolsets repos,issues,pull_requests,code_security stdio']
#     full_command = base_command + command_args
#     env = {'GITHUB_PUBLIC_ACCESS_TOKEN': settings.GITHUB_PUBLIC_ACCESS_TOKEN}

#     try:
#         process = subprocess.Popen(full_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
#                                    stderr=subprocess.PIPE, env=env, text=True)
#         stdout, stderr = process.communicate(timeout=20)
#         if stderr:
#             print(f"mcpcurl error: {stderr}")
#             return None  # Or raise an exception

#         try:
#             return json.loads(stdout)
#         except json.JSONDecodeError:
#             print(f"mcpcurl output is not valid JSON: {stdout}")
#             return stdout.strip()  # Return raw output if not JSON

#     except FileNotFoundError:
#         print(f"Error: mcpcurl not found at {mcpcurl_path}")
#         return None
#     except subprocess.TimeoutExpired:
#         print("Error: Timeout communicating with mcpcurl.")
#         return None
#     except Exception as e:
#         print(f"An unexpected error occurred while running mcpcurl: {e}")
#         return None