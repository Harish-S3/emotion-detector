import tkinter as tk
from tkinter import ttk, messagebox
import random
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
import torch
from transformers import pipeline
from datetime import datetime

# Download NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')

class ProfessionalEmotionAI:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Intelligence AI")
        self.root.geometry("900x750")
        self.root.resizable(True, True)
        
        # Initialize NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.emotion_vectors = TfidfVectorizer()
        self.history = pd.DataFrame(columns=["timestamp", "emotion", "intensity", "fortune"])
        
        # Load deep learning model
        try:
            self.emotion_classifier = pipeline(
                "text-classification", 
                model="j-hartmann/emotion-english-distilroberta-base", 
                top_k=None
            )
        except:
            self.emotion_classifier = None
            messagebox.showwarning("AI Model", "Deep learning model not available. Using enhanced NLP method.")
        
        # Set up emotion database
        self.setup_emotion_database()
        self.setup_ui()
        self.current_theme = "light"
        self.configure_styles()
        
    def setup_emotion_database(self):
        """Enhanced emotion database with psychological insights"""
        self.emotion_map = {
            'joy': {
                'keywords': ['happy', 'joy', 'excited', 'ecstatic', 'cheerful', 'delighted', 'thrilled'],
                'fortunes': [
                    "Your positive energy will attract wonderful opportunities today.",
                    "A pleasant surprise is waiting just around the corner.",
                    "Your smile will brighten someone's day more than you know."
                ],
                'emoji': "😊",
                'insight': "Joy enhances creativity and problem-solving abilities. Use this energy to tackle challenging projects!",
                'color': "#4CAF50"
            },
            'sadness': {
                'keywords': ['sad', 'depress', 'gloomy', 'down', 'unhappy', 'miserable', 'blue'],
                'fortunes': [
                    "This difficult time will pass sooner than you think.",
                    "Allow yourself to feel, then let the healing begin.",
                    "Brighter days are ahead - this cloud will lift."
                ],
                'emoji': "😔",
                'insight': "Sadness often precedes personal growth. Reflect on what this feeling might teach you.",
                'color': "#2196F3"
            },
            'fear': {
                'keywords': ['anxious', 'nervous', 'worried', 'stressed', 'tense', 'apprehensive'],
                'fortunes': [
                    "Breathe deeply - you're stronger than your anxieties.",
                    "This worry will pass, and you'll emerge wiser.",
                    "Focus on what you can control; release the rest."
                ],
                'emoji': "😰",
                'insight': "Fear activates your protective instincts. Channel this energy into preparation.",
                'color': "#FF9800"
            },
            'anger': {
                'keywords': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated'],
                'fortunes': [
                    "Channel this energy into something creative today.",
                    "Forgiveness will bring you peace sooner than anger.",
                    "Take a deep breath before responding - clarity will come."
                ],
                'emoji': "😠",
                'insight': "Anger signals boundary violations. Consider what needs protecting in your life.",
                'color': "#F44336"
            },
            'tiredness': {
                'keywords': ['tired', 'exhaust', 'fatigue', 'drain', 'weary', 'sleepy'],
                'fortunes': [
                    "Rest is not a luxury, but a necessity - honor your need for it.",
                    "Your body is asking for care - listen to it today.",
                    "A small break will recharge you more than you expect."
                ],
                'emoji': "😴",
                'insight': "Fatigue is your body's signal to slow down. Respect your natural rhythms.",
                'color': "#9C27B0"
            },
            'confusion': {
                'keywords': ['confuse', 'uncertain', 'unsure', 'lost', 'bewildered', 'perplexed'],
                'fortunes': [
                    "Clarity will come when you stop searching so hard.",
                    "Sometimes the best path reveals itself when you pause.",
                    "Trust that answers will come at the right time."
                ],
                'emoji': "😕",
                'insight': "Confusion often precedes breakthroughs. Embrace the not-knowing.",
                'color': "#00BCD4"
            },
            'neutral': {
                'keywords': ['ok', 'fine', 'alright', 'normal', 'meh', 'whatever'],
                'fortunes': [
                    "Today is a blank canvas - paint it with purpose.",
                    "Small steps lead to big changes - start now.",
                    "The ordinary moments often hold extraordinary potential."
                ],
                'emoji': "😐",
                'insight': "Neutral states provide space for reflection. What might you discover?",
                'color': "#9E9E9E"
            }
        }
        
      
        # Results Container
        results_container = ttk.Frame(main_container)
        results_container.pack(fill=tk.BOTH, expand=True)
        
        # Emotion Diagnosis
        emotion_frame = ttk.LabelFrame(results_container, text=" Emotional Diagnosis ")
        emotion_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.emotion_text = tk.StringVar(value="Your emotional state will appear here")
        ttk.Label(emotion_frame, textvariable=self.emotion_text, 
                   # Train emotion vectors
        all_text = []
        for emotion, data in self.emotion_map.items():
            all_text.extend(data['keywords'])
        self.emotion_vectors.fit(all_text)
    
    def setup_ui(self):
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(header_frame, text="Emotion Intelligence AI", 
                 font=("Segoe UI", 24, "bold")).pack(side=tk.LEFT)
        
        # Theme toggle button
        self.theme_btn = ttk.Button(header_frame, text="🌙 Dark Mode", 
                                   command=self.toggle_theme, width=12)
        self.theme_btn.pack(side=tk.RIGHT, padx=10)
        
        # Input Section
        input_frame = ttk.LabelFrame(main_container, text=" Describe Your Emotional State ")
        input_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(input_frame, text="How are you feeling right now? (We understand typos!):", 
                 font=("Segoe UI", 10)).pack(anchor="w", padx=10, pady=(10, 5))
        
        self.feeling_entry = ttk.Entry(input_frame, font=("Segoe UI", 10))
        self.feeling_entry.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.feeling_entry.insert(0, "e.g., 'Feeling anxious about my presentation but also excited'")
        
        # Action Buttons - Professional layout
        btn_frame = ttk.Frame(main_container)
        btn_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.analyze_btn = ttk.Button(btn_frame, text="Analyze Emotion", 
                                    command=self.analyze_emotion, width=15,
                                    style="Accent.TButton")
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        self.history_btn = ttk.Button(btn_frame, text="Emotion History", 
                                    command=self.show_history, width=15)
        self.history_btn.pack(side=tk.LEFT, padx=5)
        
        self.insight_btn = ttk.Button(btn_frame, text="Psychological Insights", 
                                    command=self.show_insights, width=15)
        self.insight_btn.pack(side=tk.LEFT, padx=5)
        
        # Visualization Frame
        viz_frame = ttk.LabelFrame(main_container, text=" Emotional Analysis ")
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.figure = plt.Figure(figsize=(6, 4), dpi=80)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        font=("Segoe UI", 11), wraplength=350, justify="center").pack(pady=15, padx=10)
        
        # AI Wisdom
        wisdom_frame = ttk.LabelFrame(results_container, text=" AI-Generated Wisdom ")
        wisdom_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self.wisdom_text = tk.StringVar(value="Your personalized insight will appear here")
        ttk.Label(wisdom_frame, textvariable=self.wisdom_text, 
                 wraplength=350, justify="center", font=("Segoe UI", 11)).pack(pady=15, padx=10)
    
    def configure_styles(self):
        style = ttk.Style()
        
        # Light theme configurations
        style.theme_use('clam')  # Use 'clam' theme as base for better styling control
        
        # Configure the main styles for light theme
        style.configure(".", background="#F5F7FA", foreground="black")
        style.configure("TFrame", background="#F5F7FA")
        style.configure("TLabelFrame", background="#FFFFFF", foreground="black", 
                      borderwidth=2, relief="solid")
        style.configure("TLabelFrame.Label", background="#FFFFFF", foreground="black",
                      font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background="#FFFFFF", foreground="black", 
                      font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        
        # Accent button style for light theme
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), 
                      background="#4E73DF", foreground="white")
        
        # Map button colors for light theme
        style.map("Accent.TButton",
                background=[('active', '#2E59D9'), ('pressed', '#1A3F95')],
                foreground=[('active', 'white'), ('pressed', 'white')])
        
        # Configure entry style
        style.configure("TEntry", font=("Segoe UI", 10), padding=5,
                      fieldbackground="white", foreground="black")
        
        # Set initial colors
        self.root.configure(background="#F5F7FA")
    
    def toggle_theme(self):
        style = ttk.Style()
        
        if self.current_theme == "light":
            # Dark theme configurations
            style.theme_use('clam')
            
            # Configure the main styles for dark theme
            style.configure(".", background="#2D3748", foreground="white")
            style.configure("TFrame", background="#2D3748")
            style.configure("TLabelFrame", background="#1A202C", foreground="white")
            style.configure("TLabelFrame.Label", background="#1A202C", foreground="white")
            style.configure("TLabel", background="#1A202C", foreground="white")
            style.configure("TButton", background="#4A5568", foreground="white")
            
            # Accent button style for dark theme
            style.configure("Accent.TButton", background="#4E73DF", foreground="white")
            style.map("Accent.TButton",
                    background=[('active', '#2E59D9'), ('pressed', '#1A3F95')],
                    foreground=[('active', 'white'), ('pressed', 'white')])
            
            # Configure entry style for dark theme
            style.configure("TEntry", fieldbackground="#4A5568", foreground="white")
            
            # Update root window
            self.root.configure(background="#2D3748")
            self.theme_btn.configure(text="☀️ Light Mode")
            self.current_theme = "dark"
        else:
            # Light theme configurations
            self.configure_styles()
            self.theme_btn.configure(text="🌙 Dark Mode")
            self.current_theme = "light"
        
        # Force update of all widgets
        self.update_widget_colors()
    
    def update_widget_colors(self):
        """Update colors for all widgets that need manual updating"""
        # Update button styles
        self.analyze_btn.configure(style="Accent.TButton")
        self.history_btn.configure(style="TButton")
        self.insight_btn.configure(style="TButton")
        
        # Update entry field colors
        self.feeling_entry.configure(style="TEntry")
    
    def preprocess_text(self, text):
        """Clean and normalize text for analysis"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words]  # Lemmatization
        return ' '.join(words)
    
    def detect_emotion_deep(self, text):
        """Use deep learning model to detect emotions"""
        if not self.emotion_classifier:
            return None
        
        results = self.emotion_classifier(text)
        emotions = []
        
        for result in results[0]:
            emotion = result['label'].lower()
            score = result['score']
            
            # Map to our emotion categories
            if emotion in ['joy', 'happiness']:
                mapped = 'joy'
            elif emotion in ['sadness', 'grief']:
                mapped = 'sadness'
            elif emotion in ['fear', 'nervousness']:
                mapped = 'fear'
            elif emotion in ['anger', 'annoyance']:
                mapped = 'anger'
            elif emotion in ['disgust']:
                mapped = 'anger'  # Map disgust to anger
            elif emotion in ['surprise']:
                mapped = 'joy'  # Map surprise to joy
            else:
                mapped = 'neutral'
            
            emotions.append({'emotion': mapped, 'score': score})
        
        return emotions
    
    def detect_emotion_fallback(self, text):
        """Fallback method using TF-IDF and keyword matching"""
        processed = self.preprocess_text(text)
        features = self.emotion_vectors.transform([processed])
        
        # Get scores for each emotion category
        emotion_scores = {}
        for emotion, data in self.emotion_map.items():
            score = 0
            for keyword in data['keywords']:
                try:
                    idx = self.emotion_vectors.vocabulary_[keyword]
                    score += features[0, idx]
                except KeyError:
                    pass
            emotion_scores[emotion] = score
        
        # Also consider sentiment
        sentiment = TextBlob(text).sentiment.polarity
        if sentiment > 0.3:
            emotion_scores['joy'] += 0.5
        elif sentiment < -0.3:
            emotion_scores['sadness'] += 0.5
        
        # Convert to list of dicts
        emotions = [{'emotion': k, 'score': v} for k, v in emotion_scores.items()]
        return emotions
    
    def visualize_emotions(self, emotions):
        """Create a bar chart of emotion intensities"""
        self.ax.clear()
        
        # Prepare data
        emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)[:5]
        labels = [e['emotion'].capitalize() for e in emotions]
        values = [e['score'] for e in emotions]
        colors = [self.emotion_map[e['emotion'].lower()]['color'] for e in emotions]
        
        # Create bar plot
        bars = self.ax.bar(labels, values, color=colors)
        self.ax.set_title('Emotional State Analysis', fontsize=12)
        self.ax.set_ylabel('Intensity', fontsize=10)
        self.ax.set_ylim(0, 1)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=9)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def generate_fortune(self, primary_emotion):
        """Generate a fortune based on the primary emotion"""
        fortunes = self.emotion_map[primary_emotion]['fortunes']
        return random.choice(fortunes)
    
    def analyze_emotion(self):
        """Main function to analyze the entered text"""
        text = self.feeling_entry.get()
        if not text or text.startswith("e.g.,"):
            messagebox.showwarning("Input Error", "Please describe how you're feeling")
            return
        
        # Detect emotions
        if self.emotion_classifier:
            emotions = self.detect_emotion_deep(text)
        else:
            emotions = self.detect_emotion_fallback(text)
        
        if not emotions:
            messagebox.showerror("Analysis Error", "Could not analyze your emotions. Please try again.")
            return
        
        # Get primary emotion
        primary = max(emotions, key=lambda x: x['score'])
        
        # Visualize results
        self.visualize_emotions(emotions)
        
        # Update text displays
        emoji = self.emotion_map[primary['emotion']]['emoji']
        self.emotion_text.set(
            f"Primary Emotion: {primary['emotion'].capitalize()} {emoji}\n"
            f"Confidence: {primary['score']*100:.1f}%\n\n"
            f"Full Analysis:\n" + 
            "\n".join([f"- {e['emotion'].capitalize()}: {e['score']*100:.1f}%" 
                      for e in sorted(emotions, key=lambda x: x['score'], reverse=True)])
        )
        
        # Generate and display fortune
        fortune = self.generate_fortune(primary['emotion'])
        insight = self.emotion_map[primary['emotion']]['insight']
        self.wisdom_text.set(f"{fortune}\n\nPsychological Insight:\n{insight}")
        
        # Save to history
        self.save_to_history(primary['emotion'], primary['score'], fortune)
    
    def save_to_history(self, emotion, intensity, fortune):
        """Save the current analysis to history"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = pd.DataFrame([[now, emotion, intensity, fortune]], 
                               columns=["timestamp", "emotion", "intensity", "fortune"])
        self.history = pd.concat([self.history, new_entry], ignore_index=True)
    
    def show_history(self):
        """Display the emotion history in a new window"""
        if self.history.empty:
            messagebox.showinfo("History", "No emotion history recorded yet.")
            return
        
        history_window = tk.Toplevel(self.root)
        history_window.title("Emotion History")
        history_window.geometry("800x500")
        
        # Create treeview
        tree = ttk.Treeview(history_window, columns=list(self.history.columns), show="headings")
        
        # Configure columns
        for col in self.history.columns:
            tree.heading(col, text=col.capitalize())
            tree.column(col, width=100, anchor="center")
        
        tree.column("timestamp", width=150)
        tree.column("fortune", width=300)
        
        # Add data
        for _, row in self.history.iterrows():
            tree.insert("", "end", values=list(row))
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(history_window, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add button to show trends
        btn_frame = ttk.Frame(history_window)
        btn_frame.pack(fill="x", pady=5)
        
        ttk.Button(btn_frame, text="Show Trends", command=self.show_trends).pack(side="right", padx=5)
    
    def show_trends(self):
        """Show emotion trends over time"""
        if len(self.history) < 3:
            messagebox.showinfo("Trends", "Not enough data to show trends yet.")
            return
        
        # Create trend window
        trend_window = tk.Toplevel(self.root)
        trend_window.title("Emotion Trends")
        trend_window.geometry("700x500")
        
        # Convert timestamp to datetime
        df = self.history.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Group by date and emotion
        daily = df.groupby(['date', 'emotion']).agg({'intensity': 'mean'}).unstack()
        daily.columns = daily.columns.droplevel()
        
        # Plot
        fig = plt.Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        for emotion in daily.columns:
            if emotion in self.emotion_map:
                color = self.emotion_map[emotion]['color']
                daily[emotion].plot(ax=ax, label=emotion.capitalize(), 
                                  color=color, marker='o')
        
        ax.set_title("Emotional Trends Over Time")
        ax.set_ylabel("Average Intensity")
        ax.legend()
        fig.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, trend_window)
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def show_insights(self):
        """Display psychological insights about emotions"""
        insights_window = tk.Toplevel(self.root)
        insights_window.title("Psychological Insights")
        insights_window.geometry("600x400")
        
        # Create notebook for different emotions
        notebook = ttk.Notebook(insights_window)
        
        # Add tabs for each emotion
        for emotion, data in self.emotion_map.items():
            frame = ttk.Frame(notebook, padding=10)
            notebook.add(frame, text=f"{emotion.capitalize()} {data['emoji']}")
            
            ttk.Label(frame, text=f"Understanding {emotion.capitalize()}", 
                     font=("Segoe UI", 12, "bold")).pack(pady=5, anchor="w")
            
            ttk.Label(frame, text=data['insight'], wraplength=550, 
                     font=("Segoe UI", 10)).pack(pady=10, fill=tk.X)
            
            ttk.Label(frame, text="Coping Strategies:", 
                     font=("Segoe UI", 10, "bold")).pack(pady=5, anchor="w")
            
            # Add coping strategies
            if emotion == 'joy':
                strategies = [
                    "• Share your positive energy with others",
                    "• Channel excitement into creative projects",
                    "• Practice gratitude to prolong positive states"
                ]
            elif emotion == 'sadness':
                strategies = [
                    "• Allow yourself to feel without judgment",
                    "• Connect with supportive friends or family",
                    "• Engage in comforting activities like reading or nature walks"
                ]
            elif emotion == 'fear':
                strategies = [
                    "• Practice deep breathing exercises",
                    "• Break challenges into smaller, manageable steps",
                    "• Focus on what you can control in the situation"
                ]
            elif emotion == 'anger':
                strategies = [
                    "• Take a timeout before responding",
                    "• Channel energy into physical activity",
                    "• Practice assertive communication techniques"
                ]
            elif emotion == 'tiredness':
                strategies = [
                    "• Prioritize quality sleep and rest",
                    "• Practice mindfulness to recharge",
                    "• Evaluate and adjust your workload"
                ]
            elif emotion == 'confusion':
                strategies = [
                    "• Break problems into smaller components",
                    "• Seek information from reliable sources",
                    "• Embrace uncertainty as part of the learning process"
                ]
            else:  # neutral
                strategies = [
                    "• Use this calm state for reflection",
                    "• Practice mindfulness to stay present",
                    "• Explore new interests or hobbies"
                ]
            
            for strategy in strategies:
                ttk.Label(frame, text=strategy, wraplength=550, 
                         font=("Segoe UI", 10)).pack(pady=2, anchor="w")
        
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ProfessionalEmotionAI(root)
    root.mainloop()