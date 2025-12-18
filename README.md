# 🌪️ Multi-Modal Climate Disaster Response Agent

An autonomous agentic system that combines computer vision, **LangChain** orchestration, and multimodal Generative AI to analyze disaster imagery and generate policy-compliant mitigation reports.

## 🏗️ Architecture Design

The system implements a **Neuro-Symbolic AI architecture**, using **LangChain** to bridge neural perception (computer vision) with symbolic reasoning (policy retrieval).

```mermaid
graph TD
    Input[Input Image] -->|Preprocessing| CNN[EfficientNetB4 Vision Model]
    CNN -->|Softmax| Label[Predicted Class e.g., 'Flood']
    
    subgraph "LangChain RAG Pipeline"
    Label -->|Query Construction| LC[LangChain Orchestrator]
    LC -->|Vector Search| Chroma[(ChromaDB Knowledge Base)]
    Chroma -->|Retrieved Context| Context[Climate Policy & Bio-eng Data]
    end
    
    Input --> MultiModalLLM[Google Gemini 2.5 Flash]
    Label --> MultiModalLLM
    Context --> MultiModalLLM
    
    MultiModalLLM -->|Chain of Thought| Report[Final Strategic Report]
🛠️ Tech StackOrchestration: LangChain (Context Retrieval & Chain management).Computer Vision: TensorFlow/Keras, EfficientNetB4 (Transfer Learning).Multimodal LLM: Google Gemini 2.5 Flash (via google-genai SDK).Vector Database: ChromaDB (managed via LangChain).Embeddings: HuggingFace (all-MiniLM-L6-v2).📊 Performance Metrics (Model Training)The vision component was trained on ~9,500 images across 12 disaster classes.MetricResult (Validation)NotesFinal Accuracy91.32%Achieved after 8 epochs of fine-tuning.Final Loss0.3044Strong convergence with Mixed Precision training.Training Time~20 minsOptimized using T4 GPU.Inference Latency< 2sEnd-to-end (Vision + LangChain Retrieval + Generation).🔧 Key Engineering Decisions1. Why LangChain?Decision: Used LangChain for the Retrieval Augmented Generation (RAG) pipeline.Reasoning: It decouples the retrieval logic from the model inference, allowing us to swap vector stores (Chroma) or embedding models easily without rewriting the agent core.2. EfficientNetB4 vs. ResNetDecision: Used EfficientNetB4 with Transfer Learning.Trade-off: EfficientNet offers better parameter efficiency (fewer FLOPs) than ResNet50, which is critical for lowering cloud inference costs while maintaining >90% accuracy.3. Mixed Precision Training (float16)Decision: Enabled TensorFlow mixed precision.Impact: Reduced GPU memory usage by ~40%, allowing for larger batch sizes and faster convergence.🚀 How to RunClone the repository:Bashgit clone [https://github.com/YOUR_USERNAME/Disaster-Response-Agent.git](https://github.com/YOUR_USERNAME/Disaster-Response-Agent.git)
Install dependencies:Bashpip install -r requirements.txt
Set API Key:Bashexport GEMINI_API_KEY="your_api_key_here"
Run the Agent:Bashpython climate_agent.py
