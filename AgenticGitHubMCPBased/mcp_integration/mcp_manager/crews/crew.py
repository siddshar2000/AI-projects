from crewai import Crew, Process
from ..tasks.tasks import analyze_repo_structure_task
from ..agents.agents import create_agent

def build_crew(llm, owner: str, repo: str) -> Crew:
    repo_structure_auditor= create_agent(llm)
    tasks = analyze_repo_structure_task(repo_structure_auditor, owner, repo)

    return Crew(
        agents = [repo_structure_auditor],
        tasks = tasks,
        cache=True,
        process= Process.sequential,
        verbose=True,
        chat_llm=llm,
        # Todo: Consider adding max_rpm
    )