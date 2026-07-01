# revision_checklist.py
"""
Module 1 Revision Checklist
"""

class Module1Revision:
    """Track your understanding of Module 1 concepts"""
    
    def __init__(self):
        self.topics = {
            "Chapter 1: Introduction": [
                "What is Machine Learning?",
                "Types of ML (Supervised, Unsupervised, Reinforcement)",
                "Scikit-learn ecosystem and features",
                "When to use Scikit-learn"
            ],
            "Chapter 2: Architecture & API": [
                "Scikit-learn's consistent API design",
                "Base classes (BaseEstimator, TransformerMixin)",
                "Pipeline concept",
                "Configuration through parameters"
            ],
            "Chapter 3: Datasets": [
                "Understanding X (features) and y (target)",
                "Samples vs Features",
                "Train-test split concept",
                "Feature types (numerical, categorical)"
            ],
            "Chapter 4: Workflow": [
                "fit() - Learn patterns from data",
                "predict() - Make predictions",
                "transform() - Data transformation",
                "score() - Evaluate performance",
                "fit_transform() - Combined operation"
            ],
            "Chapter 5: ML Types": [
                "Supervised Learning (Classification, Regression)",
                "Unsupervised Learning (Clustering, Dimensionality Reduction)",
                "Reinforcement Learning basics",
                "Choosing the right approach"
            ],
            "Chapter 6: Estimators vs Transformers vs Predictors": [
                "Estimators: fit() method, learn from data",
                "Transformers: transform() method, modify data",
                "Predictors: predict() method, make predictions",
                "Differences and use cases"
            ]
        }
    
    def check_understanding(self, chapter=None):
        """Print checklist for revision"""
        if chapter:
            topics = {chapter: self.topics[chapter]}
        else:
            topics = self.topics
            
        print("\n" + "="*60)
        print("📚 MODULE 1 REVISION CHECKLIST")
        print("="*60)
        
        for chapter, questions in topics.items():
            print(f"\n🔹 {chapter}")
            for q in questions:
                print(f"   ☐ {q}")
    
    def generate_quiz(self, num_questions=10):
        """Generate quiz questions"""
        # Implementation for quiz generator
        pass

# Test your knowledge
revision = Module1Revision()
revision.check_understanding()