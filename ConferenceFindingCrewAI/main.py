from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.environ['OPEN_AI_API_KEY']
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
SERPER_API_KEY = os.environ['SERPER_API_KEY']

search_tool = SerperDevTool()
venue_finder = Agent(
    role="Conference Venue Finder",
    goal="To find conference venue for {city}",
    backstory=("You are an expert conference venue finder."
    "Using online search tool to fine conference venu for provided city"
    ),
    tools=[search_tool],
    verbose=True
    )

find_venue_task = Task(
    description=(
        "Conduct a thorough search to find the best venue for the upcoming conference in provided city. "
        "Consider factors such as capacity, location, amenities, and pricing. "
        "Use online resources and databases to gather comprehensive information."
    ),
    expected_output=(
        "A list of 5 potential venues with detailed information on capacity, location, amenities, pricing, and availability."
    ),
    agent=venue_finder
)

event_planning_crew = Crew(
    tasks=[find_venue_task],
    agents=[venue_finder],
    verbose=True,
    memory=True
)

inputs = {
    "conference_name": "AI Innovations Summit",
    "requirements": "Capacity for 5000, central location, modern amenities, budget up to $50,000",
    "city": "Seattle, WA",
}

result = event_planning_crew.kickoff(inputs=inputs)

