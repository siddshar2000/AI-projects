# Agentic GitHub documentation creation product using Model Context Protocol
An application that generate an GitHub repo documentation on fly and show that in nice readable format

## Installation
1.  Clone the repository: `git clone [repository URL]`
2.  Clone offical GitHub MCP server from [github-mcp-server](https://github.com/github/github-mcp-server)
3.  `go build` inside github-mcp-server\cmd\github-mcp-server
4.  `go build -o ..\` inside github-mcp-server\mcpcurl
5.  Install dependencies: 
        `pip install crewai langchain` 
        Google API key defined in `.env` file as `GOOGLE_GEMINI_API_KEY`
        go build -o ../../mcpcurl from inside the github-mcp-server/cmd/mcpcurl
6.  Run the project: `python main.py`

## Usage
An application that generate an GitHub repo documentation on fly and show that in nice readable format

## Code structure
**github-mcp-server/**: This is the cloned GitHub repository for GitHub’s MCP Server. This will be used to run the MCP server locally.

**mcp_integration/**: It is the root directory of your Django project.

    manage.py: This file lets you run commands like starting the server, applying database changes, and more. You’ll use it to run and manage your Django project from the command line.

**mcp_integration/**: This folder contains all the project-level files and configuration settings.

    **settings.py**: It contains all the settings required to configure your Django project (e.g., installed apps, databases, and middleware).

    **urls.py**: It maps URLs to views across the entire Django project.

**mcp_manager/**: Your Django application contains the code for a specific part of your project.

    migrations/: This folder tracks changes to your database models over time.

        migrations/__init__.py: It marks the migrations folder as a Python package.

    __init__.py: It marks this folder as a Python package to import it.

    admin.py: It lets you customize how your models appear in the Django admin site.

    apps.py: It stores the app configuration for Django to recognize this app.

    models.py: It defines the database structure using Python classes.

    tests.py: It contains tests to check if your app works correctly.

    urls.py: It connects URLs to views so your app can respond to browser requests.

    views.py: It contains the functions that handle what users see and do on each page.

## Model Scorecard
### Input data


### Model used
Google's `gemini/gemini-2.0-flash`

### Sample output 
TBA