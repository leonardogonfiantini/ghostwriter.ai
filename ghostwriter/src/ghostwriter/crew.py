import json
import re
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
import litellm

@CrewBase
class PublishingHouseCrew():
    """Crew to simulate a complete publishing house"""
    
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    def __init__(self) -> None:
        # Initialize tools
        self.search_tool = SerperDevTool()
        
        # Configure local LLM with Ollama
        # This uses litellm under the hood and correctly sets up the provider.
        self.llm = LLM(
            model="ollama/deepseek-r1:14b",
            base_url="http://localhost:11434"
        )
        
        # Store dynamic chapter tasks and results
        self.chapter_tasks = []
        self.all_results = {}
    
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            tools=[self.search_tool],
            llm=self.llm,
            verbose=True
        )
    
    @agent
    def designer(self) -> Agent:
        return Agent(
            config=self.agents_config['designer'],
            llm=self.llm,
            verbose=True
        )
    
    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config['writer'],
            llm=self.llm,
            verbose=True
        )
    
    @agent
    def controller(self) -> Agent:
        return Agent(
            config=self.agents_config['controller'],
            llm=self.llm,
            verbose=True
        )
    
    @agent
    def director(self) -> Agent:
        return Agent(
            config=self.agents_config['director'],
            llm=self.llm,
            verbose=True
        )
    
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
            agent=self.researcher()
        )
    
    @task
    def design_task(self) -> Task:
        return Task(
            config=self.tasks_config['design_task'],
            agent=self.designer(),
            context=[self.research_task()]
        )
    
    def extract_chapter_count(self, design_output: str) -> int:
        """Extract number of chapters from design output"""
        # Look for patterns like "Chapter 1:", "Chapter 2:", etc.
        chapter_matches = re.findall(r'Chapter\s+(\d+):', design_output, re.IGNORECASE)
        
        if not chapter_matches:
            # Fallback: look for numbered sections
            chapter_matches = re.findall(r'(\d+)\.', design_output)
        
        if not chapter_matches:
            # Look for explicit statements about chapter count
            count_matches = re.findall(r'(\d+)\s+chapters?', design_output, re.IGNORECASE)
            if count_matches:
                return int(count_matches[-1])  # Take the last mentioned number
            
            # Default fallback
            num_chapters = 5
            print(f"⚠️ Could not extract chapter count from design. Using default: {num_chapters}")
            return num_chapters
        else:
            num_chapters = len(set(chapter_matches))  # Remove duplicates
            print(f"📖 Detected {num_chapters} chapters from design")
            return num_chapters
    
    def create_chapter_task(self, chapter_num: int, total_chapters: int) -> Task:
        """Create a single chapter writing task"""
        return Task(
            description=f"""
            Write Chapter {chapter_num} of the book following the structure defined by the designer.
            
            Use the research_context and design_context provided in the inputs to inform your writing.
            
            Requirements:
            - Follow exactly the specifications for Chapter {chapter_num} from the design
            - Maintain consistency with the established tone and style
            - Use relevant information from the research
            - Write approximately 1500-3000 words for medium length books
            - Create engaging content that flows naturally
            - End with a smooth transition (unless it's the final chapter)
            
            Chapter number: {chapter_num}
            Total chapters: {total_chapters}
            """,
            expected_output=f"""
            A complete Chapter {chapter_num} that:
            - Follows the designer's specifications exactly
            - Is stylistically consistent with the overall book
            - Includes all elements specified in the design
            - Is engaging and flows well
            - Has appropriate length for the book type
            - Maintains narrative coherence
            """,
            agent=self.writer()
        )
    
    def create_conclusion_task(self) -> Task:
        """Create the conclusion writing task"""
        return Task(
            description="""
            Write the book's conclusion that summarizes the key points covered in the
            previous chapters and provides a satisfying closure for the reader.
            
            Use the research_context, design_context, and chapters_summary from inputs.
            
            The conclusion must:
            1. Summarize the main concepts of the book
            2. Connect all chapters in an overall vision
            3. Provide final reflections on the topic
            4. Include call-to-action or suggestions for the reader
            5. Leave a lasting and positive impression
            6. Maintain the tone of voice of the rest of the book
            """,
            expected_output="""
            A conclusion of 800-1500 words that effectively closes the book,
            summarizes key points, and leaves the reader satisfied and inspired.
            """,
            agent=self.writer()
        )
    
    @task
    def control_task(self) -> Task:
        return Task(
            description="""
            Perform a complete and detailed review of all book content
            to verify quality, consistency, and correctness. This is a critical check
            to ensure excellence of the final product.
            
            Use the book_content_summary from inputs to understand the book structure.
            
            Verify the following aspects:
            
            GRAMMAR AND SYNTAX:
            1. Spelling and typing errors
            2. Grammatical and punctuation errors
            3. Sentence construction and text fluidity
            4. Correct use of verb tenses
            
            NARRATIVE CONSISTENCY:
            1. Consistency between chapters and original design
            2. Logical thread connecting all chapters
            3. Appropriate transitions between sections
            4. Maintenance of tone of voice throughout
            
            CONTENT QUALITY:
            1. Accuracy of presented information
            2. Appropriateness for target audience
            3. Completeness according to designer specifications
            4. Balance between chapters
            
            STRUCTURE AND ORGANIZATION:
            1. Adherence to original design
            2. Appropriate chapter length
            3. Logical organization of information
            4. Effectiveness of introduction and conclusion
            
            If you find errors or inconsistencies, provide:
            - Specific identification of the problem
            - Explanation of why it's problematic
            - Concrete suggestion for correction
            - Problem priority (high/medium/low)
            """,
            expected_output="""
            A detailed quality control report that includes:
            
            1. EXECUTIVE SUMMARY:
               - General quality assessment (Excellent/Good/Sufficient/Insufficient)
               - Total number of problems found by category
               - General recommendations
            
            2. DETAILED ANALYSIS:
               - Specific list of each error found with location
               - Problem categorization by type
               - Concrete suggestions for each correction
            
            3. CONSISTENCY EVALUATION:
               - Analysis of narrative consistency between chapters
               - Verification of adherence to original design
               - Assessment of overall logical thread
            
            4. RECOMMENDATIONS:
               - Priority of necessary corrections
               - Suggestions for optional improvements
               - Notes for future revisions
            
            If there are no significant problems, declare the book "APPROVED FOR PUBLICATION"
            with a brief explanation of the quality achieved.
            """,
            agent=self.controller()
        )
    
    @task
    def final_evaluation(self) -> Task:
        return Task(
            description="""
            As Editorial Director, provide a professional and comprehensive evaluation
            of the completed book, assigning a final grade based on industry-standard
            editorial criteria.
            
            Use the control_report from inputs to inform your evaluation.
            
            Evaluation criteria (each on 1-10 scale):
            
            1. WRITING QUALITY (25%)
               - Language fluidity and clarity
               - Vocabulary richness
               - Stylistic effectiveness
               - Absence of errors
            
            2. CONSISTENCY AND STRUCTURE (20%)
               - Logical content organization
               - Narrative consistency between chapters
               - Book architecture effectiveness
               - Transitions and connections
            
            3. CONTENT VALUE (25%)
               - Information depth and accuracy
               - Originality and insights
               - Relevance to target audience
               - Treatment completeness
            
            4. ENGAGEMENT AND READABILITY (15%)
               - Ability to capture attention
               - Maintaining interest
               - Language accessibility
               - Narrative rhythm
            
            5. TARGET APPROPRIATENESS (15%)
               - Tone appropriateness
               - Adequate complexity level
               - Example relevance
               - Expectation satisfaction
            
            Calculate final grade as weighted average of criteria.
            
            Final evaluation scale:
            - 9.0-10.0: Masterpiece (immediate publication, potential bestseller)
            - 8.0-8.9: Excellent (recommended publication)
            - 7.0-7.9: Good (publishable with minimal adjustments)
            - 6.0-6.9: Sufficient (moderate revisions needed)
            - 5.0-5.9: Insufficient (substantial revisions required)
            - <5.0: Unacceptable (rewriting necessary)
            """,
            expected_output="""
            A professional editorial evaluation that includes:
            
            1. EXECUTIVE SUMMARY:
               - Final grade out of 10
               - Quality category
               - Editorial recommendation
            
            2. DETAILED ANALYSIS BY CRITERIA:
               - Specific score for each criterion (1-10)
               - Detailed comments on strengths
               - Identification of improvement areas
            
            3. STRENGTHS:
               - Excellent aspects of the book
               - Distinctive and original elements
               - Features that make it competitive
            
            4. IMPROVEMENT AREAS:
               - Aspects that could be enhanced
               - Suggestions for future editions
               - Marketing considerations
            
            5. FINAL RECOMMENDATIONS:
               - Publication decision
               - Suggested marketing strategy
               - Estimated sales target
               - Notes for the author
            
            6. MARKET COMPARISON:
               - Positioning versus competition
               - Commercial potential
               - Actual target audience
            
            The evaluation must be objective, constructive, and professional,
            providing judgment that reflects publishing industry standards.
            """,
            agent=self.director()
        )
    
    def run_complete_workflow(self, inputs: dict) -> str:
        """Execute the complete book creation process using individual crews"""
        print("🔍 Starting research phase...")
        
        # Create research crew
        research_crew = Crew(
            agents=[self.researcher()],
            tasks=[self.research_task()],
            process=Process.sequential,
            verbose=True
        )
        research_result = research_crew.kickoff(inputs=inputs)
        self.all_results['research'] = research_result
        
        print("🎨 Starting design phase...")
        
        # Create design crew with research context
        design_task = self.design_task()
        design_crew = Crew(
            agents=[self.designer()],
            tasks=[design_task],
            process=Process.sequential,
            verbose=True
        )
        design_result = design_crew.kickoff(inputs=inputs)
        self.all_results['design'] = design_result
        
        # Extract chapter count from design
        num_chapters = self.extract_chapter_count(str(design_result))
        
        print(f"✍️ Starting writing phase ({num_chapters} chapters)...")
        chapter_results = []
        
        # Write each chapter using individual crews
        for i in range(1, num_chapters + 1):
            print(f"📝 Writing Chapter {i}/{num_chapters}...")
            chapter_task = self.create_chapter_task(i, num_chapters)
            
            chapter_crew = Crew(
                agents=[self.writer()],
                tasks=[chapter_task],
                process=Process.sequential,
                verbose=True
            )
            
            # Add context information to inputs
            chapter_inputs = inputs.copy()
            chapter_inputs['research_context'] = str(research_result)[:2000]  # Truncate to avoid token limits
            chapter_inputs['design_context'] = str(design_result)[:2000]
            
            chapter_result = chapter_crew.kickoff(inputs=chapter_inputs)
            chapter_results.append(chapter_result)
            self.all_results[f'chapter_{i}'] = chapter_result
        
        print("🏁 Writing conclusion...")
        conclusion_task = self.create_conclusion_task()
        conclusion_crew = Crew(
            agents=[self.writer()],
            tasks=[conclusion_task],
            process=Process.sequential,
            verbose=True
        )
        
        conclusion_inputs = inputs.copy()
        conclusion_inputs['research_context'] = str(research_result)[:1000]
        conclusion_inputs['design_context'] = str(design_result)[:1000]
        conclusion_inputs['chapters_summary'] = "Previous chapters completed: " + ", ".join([f"Chapter {i+1}" for i in range(len(chapter_results))])
        
        conclusion_result = conclusion_crew.kickoff(inputs=conclusion_inputs)
        self.all_results['conclusion'] = conclusion_result
        
        print("🔍 Starting quality control...")
        control_task = self.control_task()
        control_crew = Crew(
            agents=[self.controller()],
            tasks=[control_task],
            process=Process.sequential,
            verbose=True
        )
        
        control_inputs = inputs.copy()
        control_inputs['book_content_summary'] = f"Book has {num_chapters} chapters plus conclusion"
        
        control_result = control_crew.kickoff(inputs=control_inputs)
        self.all_results['control'] = control_result
        
        print("⭐ Final evaluation...")
        eval_task = self.final_evaluation()
        eval_crew = Crew(
            agents=[self.director()],
            tasks=[eval_task],
            process=Process.sequential,
            verbose=True
        )
        
        eval_inputs = inputs.copy()
        eval_inputs['control_report'] = str(control_result)[:1000]
        
        eval_result = eval_crew.kickoff(inputs=eval_inputs)
        self.all_results['evaluation'] = eval_result
        
        # Compile complete book
        return self._compile_book(chapter_results)
    
    def _compile_book(self, chapter_results: list) -> str:
        """Compile all parts into a complete book"""
        book_content = f"""# COMPLETE BOOK

## RESEARCH REPORT
{self.all_results.get('research', 'No research available')}

## BOOK DESIGN
{self.all_results.get('design', 'No design available')}

## BOOK CONTENT
"""
        
        # Add all chapters
        for i, chapter in enumerate(chapter_results, 1):
            book_content += f"\n### Chapter {i}\n{chapter}\n"
        
        # Add conclusion
        book_content += f"\n### Conclusion\n{self.all_results.get('conclusion', 'No conclusion available')}\n"
        
        book_content += f"""
## QUALITY CONTROL REPORT
{self.all_results.get('control', 'No control report available')}

## DIRECTOR'S FINAL EVALUATION
{self.all_results.get('evaluation', 'No evaluation available')}
"""
        
        return book_content
    
    @crew
    def crew(self) -> Crew:
        """Create the publishing house crew - simplified version"""
        return Crew(
            agents=[
                self.researcher(),
                self.designer(), 
                self.writer(),
                self.controller(),
                self.director()
            ],
            tasks=[
                self.research_task(),
                self.design_task()
            ],
            process=Process.sequential,
            verbose=True
        )