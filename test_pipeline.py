import sqlite3
import pandas as pd
from app import load_data, train_model, triage, init_db, log_ticket, recent_tickets

def run_tests():
    print("=== 1. Initializing DB and Training Model ===")
    init_db()
    df = load_data()
    print(f"Loaded {len(df)} total tickets across categories: {list(df['category'].unique())}")
    
    vec, clf, all_vecs, accuracy = train_model(df)
    print(f"Model test set accuracy: {accuracy:.2%}")
    assert accuracy > 0.50, "Accuracy should be > 50%"
    
    print("\n=== 2. Testing Standard Test Cases (Section 11) ===")
    test_cases = [
        ("VPN is not working since morning", "Network", "High"),
        ("I forgot my password", "Access", "Medium"),
        ("My laptop screen is flickering", "Hardware", "Low"),
        ("Outlook is not syncing", "Software", "Medium"),
        ("Printer is jammed", "Hardware", "Low"),
    ]
    
    for text, expected_cat, expected_pri in test_cases:
        cat, pri, conf, similar = triage(text, df, vec, clf, all_vecs)
        print(f"Input: '{text}' -> Category: {cat} (Expected: {expected_cat}), Priority: {pri}, Confidence: {conf:.1%}")
        print(f"  Top 3 similar: {similar}")
        assert cat == expected_cat, f"Expected {expected_cat}, got {cat}"
        assert pri == expected_pri, f"Expected {expected_pri}, got {pri}"
        assert len(similar) == 3, "Expected 3 similar tickets"
        log_ticket(text, cat, pri)
    
    print("\n=== 3. Testing Edge Cases ===")
    # Unseen words
    unseen_text = "Quantum flux capacitor overheated"
    cat, pri, conf, similar = triage(unseen_text, df, vec, clf, all_vecs)
    print(f"Unseen text: '{unseen_text}' -> Category: {cat}, Priority: {pri}, Confidence: {conf:.1%}")
    assert len(similar) == 3
    
    # Very long text
    long_text = "Hello support team, I am writing to report that " + "VPN fails to connect " * 20
    cat, pri, conf, similar = triage(long_text, df, vec, clf, all_vecs)
    print(f"Long text -> Category: {cat}, Priority: {pri}, Confidence: {conf:.1%}")
    assert cat == "Network"
    
    print("\n=== 4. Testing SQLite Log & Recent Tickets ===")
    recent = recent_tickets()
    print("Recent tickets in DB:")
    print(recent)
    assert len(recent) >= 5, "Database should contain at least 5 logged tickets"
    
    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
