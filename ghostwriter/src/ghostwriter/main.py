#!/usr/bin/env python3
"""
Publishing House MAS - Entry Point
Multi-Agent System for automated book creation
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def check_requirements():
    """Check if all required components are available"""
    issues = []
    
    # Check Ollama connection
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            issues.append("❌ Ollama server not responding")
    except Exception:
        issues.append("❌ Cannot connect to Ollama server")
    
    # Check environment variables
    if not os.getenv('SERPER_API_KEY'):
        issues.append("⚠️ SERPER_API_KEY not set - web search may not work")
    
    # Check config files
    config_dir = Path("src/ghostwriter/config")
    if not config_dir.exists():
        issues.append("❌ Config directory not found")
    elif not (config_dir / "agents.yaml").exists() or not (config_dir / "tasks.yaml").exists():
        issues.append("❌ Configuration files missing")
    
    return issues

def get_user_inputs():
    """Get user inputs with validation"""
    print("📚 Book Creation Configuration")
    print("-" * 30)
    
    # Topic (required)
    while True:
        topic = input("Enter the book topic: ").strip()
        if topic:
            break
        print("❌ Topic cannot be empty!")
    
    # Target audience (optional)
    target_audience = input("Enter target audience (press Enter for 'General public'): ").strip()
    if not target_audience:
        target_audience = "General public"
    
    # Book length (optional)
    print("📖 Book length options:")
    print("  - short: 3-5 chapters")
    print("  - medium: 4-8 chapters")
    print("  - long: 8-12 chapters")
    
    book_length = input("Enter desired length [medium]: ").strip().lower()
    if book_length not in ['short', 'medium', 'long']:
        book_length = 'medium'
    
    return {
        'topic': topic,
        'target_audience': target_audience,
        'book_length': book_length
    }

def save_book(content: str, topic: str) -> str:
    """Save the book to a file with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_topic = safe_topic.replace(' ', '_').lower()
    
    output_file = f"book_{safe_topic}_{timestamp}.md"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        return output_file
    except Exception as e:
        print(f"❌ Error saving book: {str(e)}")
        return None

def main():
    """Main function to start the publishing house system"""
    print("🏢 Publishing House MAS")
    print("=" * 50)
    print("Multi-Agent System for Automated Book Creation")
    print("=" * 50)
    
    # Check system requirements
    print("🔍 Checking system requirements...")
    issues = check_requirements()
    
    if issues:
        print("\n⚠️ System Issues Found:")
        for issue in issues:
            print(f"  {issue}")
        
        if any("❌" in issue for issue in issues):
            print("\n❌ Critical issues found. Please fix them before continuing.")
            return 1
        else:
            print("\n⚠️ Warning issues found, but continuing...")
    else:
        print("✅ All requirements met!")
    
    print()
    
    # Get user inputs
    try:
        inputs = get_user_inputs()
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
        return 0
    
    # Display configuration
    print(f"\n🚀 Starting book creation...")
    print(f"📖 Topic: {inputs['topic']}")
    print(f"👥 Target audience: {inputs['target_audience']}")
    print(f"📏 Length: {inputs['book_length']}")
    print("-" * 50)
    
    try:
        # Import here to avoid import errors during requirement check
        from .crew import PublishingHouseCrew
        
        # Initialize and start the crew
        print("🔧 Initializing publishing crew...")
        publishing_crew = PublishingHouseCrew()
        
        # Execute the book creation process
        print("🎬 Starting book creation workflow...")
        result = publishing_crew.run_complete_workflow(inputs=inputs)
        
        # Save the result
        output_file = save_book(str(result), inputs['topic'])
        
        # Final summary
        print("\n" + "=" * 50)
        print("✅ BOOK CREATION COMPLETED!")
        print("=" * 50)
        
        if output_file:
            print(f"💾 Book saved as: {output_file}")
            print(f"📊 File size: {os.path.getsize(output_file) / 1024:.1f} KB")
        
        print(f"📖 Topic: {inputs['topic']}")
        print(f"👥 Target: {inputs['target_audience']}")
        print(f"📏 Length: {inputs['book_length']}")
        print("\n🎉 Thank you for using Publishing House MAS!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during book creation: {str(e)}")
        print("=" * 50)
        print("🔍 Troubleshooting tips:")
        print("- Check that Ollama is running with deepseek-r1:14b model")
        print("- Verify SERPER_API_KEY environment variable")
        print("- Check config/agents.yaml and config/tasks.yaml files")
        print("- Review the full error trace above")
        
        # Print full traceback in debug mode
        if os.getenv('DEBUG'):
            import traceback
            print("\nFull error traceback:")
            traceback.print_exc()
        
        return 1

def run():
    """Entry point function for the CLI"""
    return main()

if __name__ == "__main__":
    sys.exit(main())