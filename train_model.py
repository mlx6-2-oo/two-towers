from data_handling import load_tinybert, get_embeddings
from rnn import TowerOne, TowerTwo
from training_helpers import train_towers

training_data = [
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

validation_data = [
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

# Load TinyBERT model and tokenizer
model, tokenizer = load_tinybert()

# Get embeddings for training data
embeddings_training_data = get_embeddings(training_data, model, tokenizer)
embeddings_validation_data = get_embeddings(validation_data, model, tokenizer)

# Initialize the towers
tower_one = TowerOne()
tower_two = TowerTwo()

train_towers(tower_one, tower_two, embeddings_training_data, embeddings_validation_data, num_epochs=10, margin=0.2, lr=0.001, use_wandb=True)
