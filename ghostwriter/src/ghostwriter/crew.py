from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
import os

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class PublishingHouse():
    """Publishing House crew for book creation and publishing"""
    
    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
	
	    # Initialize tools for research
    def __init__(self):
        # Solo SerperDevTool per ora, rimuoviamo WebsiteSearchTool che richiede OpenAI
        self.search_tool = SerperDevTool() if os.getenv("SERPER_API_KEY") else None
    
    # AGENTS DEFINITION
    @agent
    def story_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['story_researcher'],
            verbose=True
        )
    
    @agent
    def book_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['book_writer'],
            verbose=True
        )
    
    @agent
    def content_editor(self) -> Agent:
        return Agent(
            config=self.agents_config['content_editor'],
            verbose=True
        )
    
    @agent
    def quality_reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config['quality_reviewer'],
            verbose=True
        )
    
    @agent
    def publishing_coordinator(self) -> Agent:
        return Agent(
            config=self.agents_config['publishing_coordinator'],
            verbose=True
        )
    
    # TASKS DEFINITION
    @task
    def research_story_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_story'],
            agent=self.story_researcher(),
            output_file='story_outline.md'
        )
    
    @task
    def write_book_task(self) -> Task:
        return Task(
            config=self.tasks_config['write_book'],
            agent=self.book_writer(),
            context=[self.research_story_task()],
            output_file='manuscript.md'
        )
    
    @task
    def edit_content_task(self) -> Task:
        return Task(
            config=self.tasks_config['edit_content'],
            agent=self.content_editor(),
            context=[self.write_book_task()],
            output_file='edited_manuscript.md'
        )
    
    @task
    def review_quality_task(self) -> Task:
        return Task(
            config=self.tasks_config['review_quality'],
            agent=self.quality_reviewer(),
            context=[self.edit_content_task()],
            output_file='quality_assessment.md'
        )
    
    @task
    def coordinate_publication_task(self) -> Task:
        return Task(
            config=self.tasks_config['coordinate_publication'],
            agent=self.publishing_coordinator(),
            context=[self.research_story_task(), self.write_book_task(), 
                    self.edit_content_task(), self.review_quality_task()],
            output_file='publication_decision.md'
        )
    
    # RECURSIVE TASKS
    @task
    def revise_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config['revise_writing'],
            agent=self.book_writer(),
            context=[self.review_quality_task(), self.coordinate_publication_task()],
            output_file='manuscript_revised.md'
        )
    
    @task
    def revise_editing_task(self) -> Task:
        return Task(
            config=self.tasks_config['revise_editing'],
            agent=self.content_editor(),
            context=[self.revise_writing_task()],
            output_file='edited_manuscript_revised.md'
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the Publishing House crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,    # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead
        )