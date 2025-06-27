import json
import re
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
import litellm

@CrewBase
class PublishingHouseCrew():
    """Crew to simulate a complete publishing house with enhanced writer-controller interaction"""
    
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    def __init__(self) -> None:
        # Initialize tools
        self.search_tool = SerperDevTool()
        
        # Configure local LLM with Ollama
        self.llm = LLM(
            model="ollama/qwen3:14b",
            base_url="http://localhost:11434",
            temperature=0.4,
            seed=42
        )
        
        # Store workflow state
        self.workflow_results = {}
        self.chapter_count = 0
        self.max_revision_cycles = 3  # Maximum revision cycles per chapter
        
    # ==================== AGENTS ====================
    
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
    
    # ==================== BASE TASKS ====================
    
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
    
    @task
    def conclusion_task(self) -> Task:
        return Task(
            config=self.tasks_config['conclusion_task'],
            agent=self.writer()
        )
    
    @task
    def final_control_task(self) -> Task:
        return Task(
            config=self.tasks_config['control_task'],
            agent=self.controller()
        )
    
    @task
    def final_evaluation(self) -> Task:
        return Task(
            config=self.tasks_config['final_evaluation'],
            agent=self.director()
        )
    
    # ==================== DYNAMIC TASK CREATION ====================
    
    def create_chapter_task(self, chapter_num: int, total_chapters: int, context_tasks: list = None, revision_notes: str = None) -> Task:
        """Create a chapter writing task, optionally with revision notes"""
        
        base_description = f"""
        Write Chapter {chapter_num} of the book following the structure defined by the designer.
        
        Use the research findings and design specifications from the previous tasks to inform your writing.
        
        Requirements:
        - Follow exactly the specifications for Chapter {chapter_num} from the design
        - Maintain consistency with the established tone and style
        - Use relevant information from the research
        - Write approximately 1500-3000 words for medium length books
        - Create engaging content that flows naturally
        - End with a smooth transition (unless it's the final chapter)
        
        Chapter number: {chapter_num}
        Total chapters: {total_chapters}
        """
        
        if revision_notes:
            description = base_description + f"""
            
        REVISION NOTES FROM CONTROLLER:
        {revision_notes}
        
        Please address all the issues mentioned in the revision notes while maintaining the overall quality and structure of the chapter.
        """
        else:
            description = base_description + "\n\nMake sure to reference and build upon the research findings and follow the structural design provided."
        
        expected_output = f"""
        A complete Chapter {chapter_num} that:
        - Follows the designer's specifications exactly
        - Is stylistically consistent with the overall book
        - Includes all elements specified in the design
        - Incorporates relevant research findings
        - Is engaging and flows well
        - Has appropriate length for the book type
        - Maintains narrative coherence
        {"- Addresses all revision notes provided" if revision_notes else ""}
        """
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.writer(),
            context=context_tasks or []
        )
    
    def create_chapter_review_task(self, chapter_num: int, chapter_content: str) -> Task:
        """Create a task for the controller to review a specific chapter"""
        
        description = f"""
        Review Chapter {chapter_num} for quality, consistency, and correctness.
        
        CHAPTER CONTENT TO REVIEW:
        {chapter_content}
        
        Perform a detailed analysis focusing on:
        
        1. GRAMMAR AND SYNTAX:
           - Spelling and typing errors
           - Grammatical and punctuation errors
           - Sentence construction and text fluidity
           - Correct use of verb tenses
        
        2. CONTENT QUALITY:
           - Accuracy of presented information
           - Appropriateness for target audience
           - Completeness according to design specifications
           - Narrative flow and engagement
        
        3. CONSISTENCY:
           - Adherence to established tone and style
           - Consistency with previous chapters (if applicable)
           - Proper transitions and connections
        
        4. STRUCTURE:
           - Chapter organization and pacing
           - Appropriate length and depth
           - Clear introduction and conclusion
        
        If you find any issues, provide:
        - Specific identification of the problem with exact location
        - Explanation of why it's problematic
        - Concrete suggestion for correction
        - Priority level (HIGH/MEDIUM/LOW)
        
        Chapter number: {chapter_num}
        """
        
        expected_output = f"""
        A detailed review report for Chapter {chapter_num} that includes:
        
        1. OVERALL ASSESSMENT:
           - Quality rating (EXCELLENT/GOOD/NEEDS_IMPROVEMENT/POOR)
           - Summary of main strengths
           - Summary of main issues
        
        2. SPECIFIC ISSUES FOUND:
           - List each problem with exact location
           - Categorize by type (grammar, content, consistency, structure)
           - Provide specific correction suggestions
           - Assign priority level
        
        3. RECOMMENDATIONS:
           - Required changes (HIGH priority)
           - Suggested improvements (MEDIUM/LOW priority)
           - Overall guidance for revision
        
        4. DECISION:
           - APPROVED: Chapter is ready for publication
           - MINOR_REVISIONS: Small changes needed
           - MAJOR_REVISIONS: Significant rewriting required
           - REJECT: Chapter needs to be completely rewritten
        
        Be constructive and specific in your feedback to help the writer improve the chapter effectively.
        """
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.controller(),
            context=[]
        )
    
    # ==================== ENHANCED WORKFLOW EXECUTION ====================
    
    def run_complete_workflow(self, inputs: dict) -> str:
        """Execute the complete book creation process with enhanced writer-controller interaction"""
        try:
            # Phase 1: Research
            print("🔍 Phase 1: Research")
            research_result = self._execute_research_phase(inputs)
            
            # Phase 2: Design
            print("🎨 Phase 2: Design")
            design_result = self._execute_design_phase(inputs)
            
            # Phase 3: Enhanced Writing with immediate feedback
            print("✍️ Phase 3: Interactive Writing")
            chapters_result = self._execute_interactive_writing_phase(inputs)
            
            # Phase 4: Write Conclusion
            print("🏁 Phase 4: Conclusion")
            conclusion_result = self._execute_conclusion_phase(inputs)
            
            # Phase 5: Final Quality Control
            print("🔍 Phase 5: Final Quality Control")
            control_result = self._execute_final_control_phase(inputs)
            
            # Phase 6: Final Evaluation
            print("⭐ Phase 6: Final Evaluation")
            evaluation_result = self._execute_evaluation_phase(inputs)
            
            # Compile final book
            return self._compile_final_book()
            
        except Exception as e:
            print(f"❌ Error during workflow execution: {str(e)}")
            raise
    
    def _execute_research_phase(self, inputs: dict) -> str:
        """Execute research phase"""
        research_task = self.research_task()
        
        research_crew = Crew(
            agents=[self.researcher()],
            tasks=[research_task],
            process=Process.sequential,
            verbose=True
        )
        
        result = research_crew.kickoff(inputs=inputs)
        self.workflow_results['research'] = result
        return result
    
    def _execute_design_phase(self, inputs: dict) -> str:
        """Execute design phase"""
        design_task = self.design_task()
        
        design_crew = Crew(
            agents=[self.designer()],
            tasks=[design_task],
            process=Process.sequential,
            verbose=True
        )
        
        result = design_crew.kickoff(inputs=inputs)
        self.workflow_results['design'] = result
        
        # Extract chapter count
        self.chapter_count = self._extract_chapter_count(str(result))
        print(f"📖 Detected {self.chapter_count} chapters from design")
        
        return result
    
    def _execute_interactive_writing_phase(self, inputs: dict) -> list:
        """Execute interactive writing phase with immediate controller feedback"""
        # Get context tasks for chapters
        research_task = self.research_task()
        design_task = self.design_task()
        context_tasks = [research_task, design_task]
        
        chapter_results = []
        
        for i in range(1, self.chapter_count + 1):
            print(f"\n📝 === WRITING CHAPTER {i}/{self.chapter_count} ===")
            
            # Interactive writing and review cycle
            final_chapter = self._write_and_review_chapter(
                chapter_num=i,
                total_chapters=self.chapter_count,
                context_tasks=context_tasks,
                inputs=inputs
            )
            
            chapter_results.append(final_chapter)
            self.workflow_results[f'chapter_{i}'] = final_chapter
            
            print(f"✅ Chapter {i} completed and approved!")
        
        return chapter_results
    
    def _write_and_review_chapter(self, chapter_num: int, total_chapters: int, context_tasks: list, inputs: dict) -> str:
        """Write a chapter with immediate controller feedback and revision cycles"""
        
        revision_cycle = 0
        revision_notes = None
        
        while revision_cycle < self.max_revision_cycles:
            revision_cycle += 1
            
            # Write/revise the chapter
            if revision_cycle == 1:
                print(f"📝 Writing initial draft of Chapter {chapter_num}...")
            else:
                print(f"🔄 Revision cycle {revision_cycle-1} for Chapter {chapter_num}...")
            
            chapter_task = self.create_chapter_task(
                chapter_num=chapter_num,
                total_chapters=total_chapters,
                context_tasks=context_tasks,
                revision_notes=revision_notes
            )
            
            chapter_crew = Crew(
                agents=[self.writer()],
                tasks=[chapter_task],
                process=Process.sequential,
                verbose=True
            )
            
            chapter_result = chapter_crew.kickoff(inputs=inputs)
            chapter_content = str(chapter_result)
            
            # Controller reviews the chapter
            print(f"🔍 Controller reviewing Chapter {chapter_num}...")
            
            review_task = self.create_chapter_review_task(chapter_num, chapter_content)
            
            review_crew = Crew(
                agents=[self.controller()],
                tasks=[review_task],
                process=Process.sequential,
                verbose=True
            )
            
            review_result = review_crew.kickoff(inputs=inputs)
            review_content = str(review_result)
            
            # Parse the review decision
            decision = self._parse_review_decision(review_content)
            
            print(f"📊 Review Decision: {decision}")
            
            if decision == "APPROVED":
                print(f"✅ Chapter {chapter_num} approved on cycle {revision_cycle}!")
                return chapter_content
            elif decision == "MINOR_REVISIONS" and revision_cycle >= 2:
                print(f"⚠️ Chapter {chapter_num} has minor issues but reached revision limit. Accepting...")
                return chapter_content
            elif decision == "REJECT" and revision_cycle >= self.max_revision_cycles:
                print(f"❌ Chapter {chapter_num} still has major issues after {self.max_revision_cycles} cycles. Accepting current version...")
                return chapter_content
            else:
                # Extract revision notes for next cycle
                revision_notes = self._extract_revision_notes(review_content)
                print(f"🔄 Chapter {chapter_num} needs revision. Cycle {revision_cycle}/{self.max_revision_cycles}")
                
                # Store the review for reference
                self.workflow_results[f'chapter_{chapter_num}_review_{revision_cycle}'] = review_content
        
        # If we've exhausted all revision cycles, return the last version
        print(f"⏰ Maximum revision cycles reached for Chapter {chapter_num}. Using final version.")
        return chapter_content
    
    def _execute_conclusion_phase(self, inputs: dict) -> str:
        """Execute conclusion writing phase"""
        # Build context with all previous work
        all_previous_tasks = [
            self.research_task(),
            self.design_task()
        ]
        
        # Add approved chapters to context
        for i in range(1, self.chapter_count + 1):
            chapter_task = self.create_chapter_task(i, self.chapter_count)
            all_previous_tasks.append(chapter_task)
        
        conclusion_task = self.conclusion_task()
        conclusion_task.context = all_previous_tasks
        
        conclusion_crew = Crew(
            agents=[self.writer()],
            tasks=[conclusion_task],
            process=Process.sequential,
            verbose=True
        )
        
        result = conclusion_crew.kickoff(inputs=inputs)
        self.workflow_results['conclusion'] = result
        return result
    
    def _execute_final_control_phase(self, inputs: dict) -> str:
        """Execute final quality control phase on the complete book"""
        # Build context with all completed content
        all_tasks = [
            self.research_task(),
            self.design_task(),
            self.conclusion_task()
        ]
        
        # Add all approved chapters
        for i in range(1, self.chapter_count + 1):
            chapter_task = self.create_chapter_task(i, self.chapter_count)
            all_tasks.append(chapter_task)
        
        control_task = self.final_control_task()
        control_task.context = all_tasks
        
        control_crew = Crew(
            agents=[self.controller()],
            tasks=[control_task],
            process=Process.sequential,
            verbose=True
        )
        
        result = control_crew.kickoff(inputs=inputs)
        self.workflow_results['final_control'] = result
        return result
    
    def _execute_evaluation_phase(self, inputs: dict) -> str:
        """Execute final evaluation phase"""
        # Build context with all tasks including final control
        all_tasks = [
            self.research_task(),
            self.design_task(),
            self.conclusion_task(),
            self.final_control_task()
        ]
        
        # Add all chapters
        for i in range(1, self.chapter_count + 1):
            chapter_task = self.create_chapter_task(i, self.chapter_count)
            all_tasks.append(chapter_task)
        
        eval_task = self.final_evaluation()
        eval_task.context = all_tasks
        
        eval_crew = Crew(
            agents=[self.director()],
            tasks=[eval_task],
            process=Process.sequential,
            verbose=True
        )
        
        result = eval_crew.kickoff(inputs=inputs)
        self.workflow_results['evaluation'] = result
        return result
    
    # ==================== UTILITY METHODS ====================
    
    def _extract_chapter_count(self, design_output: str) -> int:
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
                return int(count_matches[-1])
            
            # Default fallback
            print("⚠️ Could not extract chapter count from design. Using default: 5")
            return 5
        else:
            return len(set(chapter_matches))
    
    def _parse_review_decision(self, review_content: str) -> str:
        """Extract the review decision from controller's feedback"""
        # Look for decision keywords
        content_upper = review_content.upper()
        
        if "APPROVED" in content_upper:
            return "APPROVED"
        elif "MINOR_REVISIONS" in content_upper or "MINOR REVISIONS" in content_upper:
            return "MINOR_REVISIONS"
        elif "MAJOR_REVISIONS" in content_upper or "MAJOR REVISIONS" in content_upper:
            return "MAJOR_REVISIONS"
        elif "REJECT" in content_upper:
            return "REJECT"
        else:
            # Default to minor revisions if unclear
            return "MINOR_REVISIONS"
    
    def _extract_revision_notes(self, review_content: str) -> str:
        """Extract specific revision notes from controller's feedback"""
        # Look for sections with recommendations or specific issues
        lines = review_content.split('\n')
        revision_notes = []
        
        collecting_notes = False
        for line in lines:
            line = line.strip()
            
            # Start collecting after keywords
            if any(keyword in line.upper() for keyword in ['SPECIFIC ISSUES', 'RECOMMENDATIONS', 'REQUIRED CHANGES', 'CORRECTIONS']):
                collecting_notes = True
                continue
            
            # Stop collecting at next major section
            if collecting_notes and line.upper().startswith(('OVERALL', 'DECISION', 'SUMMARY')):
                break
            
            # Collect meaningful lines
            if collecting_notes and line and not line.startswith('='):
                revision_notes.append(line)
        
        if revision_notes:
            return '\n'.join(revision_notes)
        else:
            # Fallback: return the whole review
            return review_content
    
    def _compile_final_book(self) -> str:
        """Compile all parts into a complete book"""
        book_content = f"""# COMPLETE BOOK

## RESEARCH REPORT
{self.workflow_results.get('research', 'No research available')}

## BOOK DESIGN
{self.workflow_results.get('design', 'No design available')}

## BOOK CONTENT
"""
        
        # Add all chapters
        for i in range(1, self.chapter_count + 1):
            chapter_content = self.workflow_results.get(f'chapter_{i}', f'Chapter {i} not available')
            book_content += f"\n### Chapter {i}\n{chapter_content}\n"
        
        # Add conclusion
        book_content += f"\n### Conclusion\n{self.workflow_results.get('conclusion', 'No conclusion available')}\n"
        
        # Add revision history summary
        book_content += f"\n## REVISION HISTORY\n"
        revision_summary = self._generate_revision_summary()
        book_content += revision_summary
        
        book_content += f"""
## FINAL QUALITY CONTROL REPORT
{self.workflow_results.get('final_control', 'No final control report available')}

## DIRECTOR'S FINAL EVALUATION
{self.workflow_results.get('evaluation', 'No evaluation available')}
"""
        
        return book_content
    
    def _generate_revision_summary(self) -> str:
        """Generate a summary of the revision process"""
        summary = "### Revision Process Summary\n\n"
        
        for i in range(1, self.chapter_count + 1):
            chapter_revisions = []
            for cycle in range(1, self.max_revision_cycles + 1):
                review_key = f'chapter_{i}_review_{cycle}'
                if review_key in self.workflow_results:
                    chapter_revisions.append(f"  - Cycle {cycle}: Review completed")
            
            if chapter_revisions:
                summary += f"Chapter {i}:\n"
                summary += "\n".join(chapter_revisions) + "\n\n"
            else:
                summary += f"Chapter {i}: Approved on first draft\n\n"
        
        return summary
    
    # ==================== SIMPLIFIED CREW (for compatibility) ====================
    
    @crew
    def crew(self) -> Crew:
        """Create a simplified crew - mainly for testing"""
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