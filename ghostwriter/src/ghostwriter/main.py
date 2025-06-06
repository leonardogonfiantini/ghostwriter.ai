#!/usr/bin/env python
import sys
import os
import re
from dotenv import load_dotenv
from crewai import LLM
from crew import PublishingHouse

# Carica le variabili d'ambiente
load_dotenv()

def setup_llama_llm():
    """
    Configura il modello LLM per Llama locale
    """
    return LLM(
        model="ollama/llama3.1",  # Sostituisci con il tuo modello
        base_url="http://localhost:11434"
    )

def count_words_in_file(filepath):
    """Conta le parole in un file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # Rimuovi markdown e conta solo le parole del contenuto
            content = re.sub(r'[#*`\-\[\]()]', '', content)
            words = len(content.split())
            return words
    except FileNotFoundError:
        return 0

def run_recursive_publishing():
    """
    Esegue il processo di pubblicazione ricorsivo con controllo qualità.
    """
    
    # Configura il modello LLM
    llm = setup_llama_llm()

    target_word_count = 50000  # Modifica questo valore
    topic = 'A child storybook about a bear and a bee'
    
    inputs = {
        'topic': topic,
        'word_count': str(target_word_count)
    }
    
    publishing_crew = PublishingHouse()
   
    # Configura il crew per usare il modello locale
    crew_instance = publishing_crew.crew()
    
    # Assegna il modello LLM a tutti gli agenti
    for agent in crew_instance.agents:
        agent.llm = llm

    max_iterations = 3  # Massimo numero di revisioni
    current_iteration = 0
    
    print(f"🚀 Avvio processo di pubblicazione per: {topic}")
    print(f"📊 Target parole: {target_word_count}")
    print("=" * 60)
    
    while current_iteration < max_iterations:
        current_iteration += 1
        print(f"\n🔄 ITERAZIONE {current_iteration}/{max_iterations}")
        
        try:
            # Esegui il processo principale
            result = publishing_crew.crew().kickoff(inputs=inputs)
            
            # Verifica il risultato
            decision_file = 'publication_decision.md'
            manuscript_file = 'edited_manuscript.md'
            
            if os.path.exists(decision_file):
                with open(decision_file, 'r', encoding='utf-8') as f:
                    decision_content = f.read()
                
                # Controlla se è stato approvato
                if 'PUBLISH' in decision_content.upper() and 'DO NOT PUBLISH' not in decision_content.upper():
                    print("✅ MANOSCRITTO APPROVATO PER LA PUBBLICAZIONE!")
                    
                    # Verifica finale del conteggio parole
                    if os.path.exists(manuscript_file):
                        actual_words = count_words_in_file(manuscript_file)
                        print(f"📝 Conteggio parole finale: {actual_words}/{target_word_count}")
                        
                        if abs(actual_words - target_word_count) <= 50:
                            print("✅ Conteggio parole conforme!")
                            return result
                        else:
                            print("❌ Conteggio parole non conforme, richiesta revisione...")
                    break
                
                elif 'REVISE' in decision_content.upper():
                    print("🔄 Richiesta revisione, avvio processo correttivo...")
                    
                    # Esegui tasks di revisione
                    if current_iteration < max_iterations:
                        print("📝 Esecuzione revisione scrittura...")
                        revision_crew = PublishingHouse()
                        revision_crew.revise_writing_task().execute_sync()
                        
                        print("✏️ Esecuzione revisione editing...")
                        revision_crew.revise_editing_task().execute_sync()
                        
                        print("🔍 Ri-esecuzione quality review...")
                        revision_crew.review_quality_task().execute_sync()
                        
                        continue
                    else:
                        print("❌ Massimo numero di revisioni raggiunto")
                        break
                
                else:
                    print("❌ MANOSCRITTO RIFIUTATO")
                    print("Motivi:", decision_content[:200] + "...")
                    break
            
            else:
                print("❌ File di decisione non trovato")
                break
                        
        except Exception as e:
            print(f"❌ Errore durante l'iterazione {current_iteration}: {e}")
            if current_iteration >= max_iterations:
                break
            continue
    
    print(f"\n📊 PROCESSO COMPLETATO DOPO {current_iteration} ITERAZIONI")
    return result if 'result' in locals() else None

def run():
    """
    Run the publishing house crew con processo ricorsivo.
    """
    return run_recursive_publishing()

def simple_run():
    """
    Esecuzione semplice senza ricorsività (per test)
    """
    inputs = {
        'topic': 'Artificial Intelligence and Machine Learning',
        'word_count': '5000'
    }
    
    publishing_crew = PublishingHouse()
    result = publishing_crew.crew().kickoff(inputs=inputs)
    
    print("Publishing process completed!")
    print("Final result:")
    print(result)
    return result

def train():
    """
    Train the crew for 10 iterations (optional).
    """
    inputs = {
        'topic': 'Artificial Intelligence',
        'word_count': '30000'
    }
    
    try:
        publishing_crew = PublishingHouse()
        publishing_crew.crew().train(n_iterations=10, inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task (optional).
    """
    inputs = {
        'topic': 'Blockchain Technology',
        'word_count': '40000'
    }
    
    try:
        publishing_crew = PublishingHouse()
        publishing_crew.crew().replay(task_id='research_story_task', inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            train()
        elif sys.argv[1] == 'replay':
            replay()
        elif sys.argv[1] == 'simple':
            simple_run()
        else:
            print("Comandi disponibili: train, replay, simple")
            print("Per esecuzione ricorsiva (default): python main.py")
    else:
        run()