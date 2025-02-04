from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm

def load_tinybert():
    """Load TinyBERT model and tokenizer from Hugging Face."""
    model_name = "huawei-noah/TinyBERT_General_4L_312D"
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading model from {model_name}...")
    model = AutoModel.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    return model, tokenizer

def get_embeddings(data, model, tokenizer, batch_size=32):
    """Convert text data into embeddings using TinyBERT with batch processing."""
    embeddings = []
    device = model.device  # Get the device model is on
    
    # Process data in batches with progress bar
    for i in tqdm(range(0, len(data), batch_size), desc="Generating embeddings"):
        batch = data[i:i + batch_size]
        
        # Prepare batch inputs
        queries = [item[0] for item in batch]
        relevant_passages = [item[1] for item in batch]
        irrelevant_passages = [item[2] for item in batch]
        
        # Get embeddings for each type in batch
        with torch.no_grad():
            # Process queries
            query_tokens = tokenizer(queries, return_tensors="pt", padding=True, truncation=True).to(device)
            query_outputs = model(**query_tokens)
            query_embeddings = query_outputs.last_hidden_state[:, 0, :].cpu()  # Move back to CPU
            
            # Process relevant passages
            relevant_tokens = tokenizer(relevant_passages, return_tensors="pt", padding=True, truncation=True).to(device)
            relevant_outputs = model(**relevant_tokens)
            relevant_embeddings = relevant_outputs.last_hidden_state[:, 0, :].cpu()
            
            # Process irrelevant passages
            irrelevant_tokens = tokenizer(irrelevant_passages, return_tensors="pt", padding=True, truncation=True).to(device)
            irrelevant_outputs = model(**irrelevant_tokens)
            irrelevant_embeddings = irrelevant_outputs.last_hidden_state[:, 0, :].cpu()
        
        # Add batch results to embeddings list
        for j in range(len(batch)):
            embeddings.append((
                query_embeddings[j:j+1],
                relevant_embeddings[j:j+1],
                irrelevant_embeddings[j:j+1]
            ))
    
    return embeddings

mock_training_data = [
    ("What is a good python test library", "Why Pytest is the best python testing library", "Dogs go wild on boozy night out with kitten"),
    ("How to debug memory leaks", "Advanced memory leak detection in Python", "Top 10 pizza recipes for beginners"),
    ("Best practices for API design", "RESTful API design patterns and principles", "Ancient Egyptian burial customs"),
    ("Python async programming guide", "Understanding asyncio and coroutines", "The history of medieval warfare"),
    ("Docker container security", "Securing Docker containers in production", "Best houseplants for your bedroom"),
    ("Machine learning deployment", "MLOps best practices and deployment strategies", "Traditional Irish folk songs"),
    ("Kubernetes vs Docker Swarm", "Container orchestration platforms compared", "Making the perfect sourdough bread"),
    ("GraphQL vs REST APIs", "Modern API architectural patterns", "Top fishing spots in Canada"),
    ("Microservices architecture patterns", "Building resilient microservices", "Ancient Roman cooking techniques"),
    ("Database indexing strategies", "Optimizing database performance", "Training your pet parrot"),
    ("CI/CD pipeline setup", "Automated deployment workflow guide", "History of Renaissance art"),
    ("Web security best practices", "Preventing common web vulnerabilities", "Growing tomatoes in your garden"),
    ("Cloud cost optimization", "Reducing AWS infrastructure costs", "Best hiking trails in Colorado"),
    ("Serverless architecture guide", "AWS Lambda and serverless computing", "Vintage car restoration tips"),
    ("Data streaming with Kafka", "Real-time data processing patterns", "Traditional Japanese tea ceremonies"),
    ("MongoDB vs PostgreSQL", "Choosing the right database for your project", "Basic knitting patterns for beginners")
]

mock_validation_data = [
    ("How to optimize Python code", "Advanced Python performance tuning guide", "Making homemade ice cream"),
    ("Git branching strategies", "Git workflow best practices for teams", "Popular bird watching locations"),
    ("SQL query optimization", "Database query performance tuning", "Beginner's guide to origami"),
    ("Load balancing techniques", "Scaling web applications with load balancers", "History of jazz music"),
    ("Monitoring microservices", "Observability patterns for distributed systems", "Indoor herb garden tips"),
    ("OAuth2 implementation guide", "Secure authentication protocols explained", "Famous French painters"),
    ("Redis caching strategies", "Implementing efficient caching layers", "Caring for indoor succulents"),
    ("Elasticsearch best practices", "Search engine optimization patterns", "Traditional Thai cooking"),
    ("Message queue architectures", "Asynchronous communication patterns", "Rock climbing for beginners"),
    ("Infrastructure as code", "Terraform and cloud provisioning", "Making artisanal cheese")
]
