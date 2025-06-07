#!/usr/bin/env python3
"""
Publishing House MAS - Entry Point
Multi-Agent System for automated book creation
"""

import os
from .crew import PublishingHouseCrew

def run():
    """
    Entry point function for the CLI
    """
    return main()

def main():
    """
    Main function to start the publishing house system
    """
    print("🏢 Welcome to Publishing House MAS")
    print("=" * 50)
    
    # User input
    topic = input("📚 Enter the book topic: ").strip()
    if not topic:
        print("❌ Invalid topic!")
        return
    
    target_audience = input("👥 Enter target audience (optional): ").strip()
    if not target_audience:
        target_audience = "General public"
    
    book_length = input("📖 Enter desired length (short/medium/long) [medium]: ").strip().lower()
    if book_length not in ['short', 'medium', 'long']:
        book_length = 'medium'
    
    print(f"\n🚀 Starting book creation on: '{topic}'")
    print(f"🎯 Target audience: {target_audience}")
    print(f"📏 Length: {book_length}")
    print("-" * 50)
    
    try:
        # Initialize and start the crew
        publishing_crew = PublishingHouseCrew()
        
        # Input for the crew
        inputs = {
            'topic': topic,
            'target_audience': target_audience,
            'book_length': book_length
        }
        
        # Execute the book creation process
        result = publishing_crew.run_complete_workflow(inputs=inputs)
        
        print("\n" + "=" * 50)
        print("✅ BOOK COMPLETED!")
        print("=" * 50)
        print(result)
        
        # Save the result
        output_file = f"book_{topic.replace(' ', '_').lower()}.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(str(result))
        
        print(f"\n💾 Book saved as: {output_file}")
        
    except Exception as e:
        print(f"❌ Error during book creation: {str(e)}")
        import traceback
        print("Full error traceback:")
        traceback.print_exc()
        return

if __name__ == "__main__":
    # Configure environment variables if needed
    if not os.getenv('SERPER_API_KEY'):
        print("⚠️ Warning: SERPER_API_KEY not set. Web search may not work.")
        print("Please set your Serper API key as an environment variable.")
        print("You can get one at: https://serper.dev/")
    
    main()