# A Scalable Content-Based Recommendation System for Sri Lankan Tourism

A PySpark-based recommendation system that provides personalized accommodation recommendations using real-world Airbnb listings data from Sri Lanka.

## 📋 Overview

This project implements a sophisticated content-based recommendation system leveraging Apache Spark's distributed computing capabilities for scalable data processing. The system is specifically designed to analyze Airbnb listing attributes from Sri Lanka's tourism market and generate highly relevant accommodation recommendations based on advanced similarity metrics.

The recommendation engine processes real-world data to help tourists discover accommodations that match their preferences, while also providing valuable insights for tourism platforms and accommodation providers. By utilizing PySpark's MLlib and distributed computing framework, the system can efficiently handle large-scale datasets and provide recommendations in near real-time.

## 🎯 Objective

Build a production-ready, scalable data processing pipeline that:

- **Data Processing**: Ingests and processes real-world Airbnb dataset containing thousands of listings across Sri Lanka
- **Feature Engineering**: Transforms raw listing attributes into meaningful numerical representations using advanced ML techniques
- **Intelligent Recommendations**: Provides content-based recommendations using cosine similarity algorithms to match user preferences
- **Performance Evaluation**: Implements comprehensive evaluation framework using Precision@K metrics to validate recommendation quality
- **Scalability**: Designed to handle growing datasets with millions of listings through distributed computing
- **Reproducibility**: Ensures consistent results across different environments and execution contexts

## 🚀 Features

### Core Capabilities

- **Distributed Processing**: 
  - Built on Apache Spark 3.x for horizontal scalability
  - Handles datasets with millions of records across multiple nodes
  - Optimized for both single-machine and cluster deployments
  - Efficient memory management with lazy evaluation

- **Advanced Feature Engineering**: 
  - Multi-stage ML pipeline with 5+ transformation stages
  - String indexing for categorical data conversion
  - One-hot encoding for binary feature representation
  - Vector assembly for unified feature representation
  - L2 normalization for cosine similarity optimization
  
- **Content-Based Filtering**: 
  - Cosine similarity-based recommendation algorithm
  - Configurable similarity thresholds
  - Support for multiple feature types (categorical, numerical, text)
  - Efficient similarity computation using dot products on normalized vectors
  
- **Comprehensive Model Evaluation**: 
  - Offline validation framework with Precision@K metrics
  - Sample-based testing methodology
  - Relevance proxy using room type similarity
  - Statistical performance analysis
  
- **Scalable Architecture**: 
  - Modular design for easy extension
  - Optimized Spark SQL operations
  - Broadcast variables for efficient data sharing
  - Configurable parallelism levels

- **Production-Ready Code**:
  - Proper resource management and cleanup
  - Error handling and logging
  - Configurable parameters
  - Well-documented functions

## 📊 Dataset

### Source Information

**Dataset**: Sri Lanka Airbnb Listings (`sri_lanka_airbnb.csv`)

**Description**: A comprehensive collection of Airbnb listings from across Sri Lanka, encompassing various types of accommodations from budget-friendly private rooms to luxury entire properties. The dataset represents real-world tourism data and includes listings from major cities, coastal areas, hill country, and cultural sites throughout Sri Lanka.

### Dataset Characteristics

- **Size**: Multiple thousand records
- **Format**: CSV (Comma-Separated Values)
- **Encoding**: UTF-8
- **Data Quality**: Cleaned and preprocessed for analysis

### Key Attributes Used

| Attribute | Type | Description | Example Values |
|-----------|------|-------------|----------------|
| `name` | String | Unique name/title of the listing | "Cozy Apartment in Colombo", "Beach Villa Galle" |
| `room_type` | Categorical | Type of accommodation offering | "Private room", "Entire home/apt", "Shared room", "Hotel room" |
| `star_rating` | Numerical | Average rating from guest reviews (0-5 scale) | 4.5, 4.8, 5.0 |
| `number_of_guests` | Integer | Maximum guest capacity | 1, 2, 4, 6, 8+ |

### Data Preprocessing

The raw dataset undergoes several preprocessing steps:

1. **Schema Inference**: Automatic data type detection
2. **Column Selection**: Relevant features are extracted
3. **Null Handling**: Missing values are filtered or imputed
4. **Data Validation**: Ensuring data quality and consistency
5. **Feature Normalization**: Scaling numerical features for uniform contribution

### Future Dataset Enhancements

Potential additional attributes that could improve recommendations:
- Location coordinates (latitude, longitude)
- Price per night
- Amenities list (WiFi, parking, pool, etc.)
- Host information (superhost status, response rate)
- Availability calendar
- Review text sentiment analysis
- Property images for visual similarity

## 🛠️ Technology Stack

### Core Technologies

- **Python 3.7+**
  - Primary programming language
  - Rich ecosystem for data science
  - Extensive library support
  
- **Apache Spark 3.x (PySpark)**
  - Distributed data processing framework
  - In-memory computation for speed
  - Resilient Distributed Datasets (RDDs)
  - Spark SQL for structured data processing
  - Optimized query execution engine
  
- **Spark MLlib**
  - Machine learning library
  - Pipeline API for workflow management
  - Feature transformers (StringIndexer, OneHotEncoder, VectorAssembler)
  - Normalizer for vector normalization
  - Scalable ML algorithms

- **Jupyter Notebook**
  - Interactive development environment
  - Cell-based execution model
  - Rich visualization support
  - Markdown documentation integration
  - Easy debugging and experimentation

### Development Tools

- **PySpark SQL**: Structured data querying and manipulation
- **PySpark DataFrame API**: High-level data manipulation
- **Python Standard Library**: Built-in utilities and functions

### System Requirements

- **Memory**: Minimum 4GB RAM (8GB+ recommended for larger datasets)
- **Processor**: Multi-core CPU (4+ cores recommended)
- **Storage**: 2GB+ free disk space
- **Operating System**: macOS, Linux, or Windows with appropriate Spark setup

## 📦 Installation

### Prerequisites

Before setting up the project, ensure you have the following installed:

- **Python 3.7 or higher** - Download from [python.org](https://www.python.org/downloads/)
- **pip package manager** - Usually comes with Python installation
- **Java 8 or 11** - Required for Apache Spark (Download from [Oracle](https://www.oracle.com/java/technologies/downloads/) or use OpenJDK)
- **4GB+ RAM** - For running Spark locally

### Detailed Setup Instructions

#### Step 1: Verify Java Installation

```bash
java -version
```

You should see Java version 8 or 11. If not installed, download and install Java first.

#### Step 2: Clone or Download the Repository

```bash
# If using git
git clone <repository-url>
cd BigData_Project

# Or download and extract the ZIP file
```

#### Step 3: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### Step 4: Install Required Dependencies

```bash
# Install PySpark
pip install pyspark

# Optional: Install additional useful packages
pip install jupyter notebook pandas numpy
```

#### Step 5: Verify Installation

```bash
# Test PySpark installation
python -c "import pyspark; print(pyspark.__version__)"
```

#### Step 6: Prepare Dataset

Ensure the dataset file `sri_lanka_airbnb.csv` is in the project root directory:

```
BigData_Project/
├── A_Scalable_Content_Based_Recommendation_System_for_Sri_Lankan_Tourism.ipynb
├── sri_lanka_airbnb.csv  ← Dataset should be here
├── README.md
└── ...
```

### Troubleshooting Installation

**Issue**: Java not found
- **Solution**: Install Java 8 or 11 and set JAVA_HOME environment variable

**Issue**: PySpark import error
- **Solution**: Ensure Python version is 3.7+ and pip installation completed successfully

**Issue**: Memory errors when running Spark
- **Solution**: Increase Spark memory allocation in the notebook or reduce dataset size for testing

## 🏃 Usage

### Running the Notebook

#### Method 1: Using Jupyter Notebook

1. **Start Jupyter Notebook server**:

```bash
jupyter notebook
```

This will open a browser window with the Jupyter file explorer.

2. **Navigate to and open the notebook**:
   - Click on `A_Scalable_Content_Based_Recommendation_System_for_Sri_Lankan_Tourism.ipynb`

3. **Execute the notebook**:
   - Run cells sequentially: Press `Shift + Enter` on each cell
   - Or run all cells: Menu → Cell → Run All

#### Method 2: Using Jupyter Lab

```bash
jupyter lab A_Scalable_Content_Based_Recommendation_System_for_Sri_Lankan_Tourism.ipynb
```

### Notebook Execution Flow

The notebook follows this structured workflow:

**Section 1: Environment Setup (Cells 1-2)**
- Initializes SparkSession with optimized configurations
- Imports necessary libraries (pyspark.sql, pyspark.ml)
- Sets up logging and display options

**Section 2: Data Ingestion & Preprocessing (Cells 3-5)**
- Loads CSV data with automatic schema inference
- Displays dataset structure and sample records
- Selects relevant columns: name, room_type, star_rating, number_of_guests
- Filters out null values to ensure data quality

**Section 3: Feature Engineering Pipeline (Cells 6-8)**
- **StringIndexer**: Converts room_type to numerical indices
- **OneHotEncoder**: Creates binary vectors for categorical features
- **VectorAssembler**: Combines all features into single vector column
- **Normalizer**: Applies L2 normalization for cosine similarity
- Fits the pipeline and transforms the dataset

**Section 4: Recommendation Generation (Cells 9-11)**
- Implements `recommend_spark()` function
- Calculates cosine similarity using dot product of normalized vectors
- Ranks items by similarity score
- Returns top-K most similar items (excludes the query item itself)

**Section 5: Model Evaluation (Cells 12-14)**
- Implements `evaluate_precision_at_k()` function
- Tests on random sample of items (default: 20 items)
- Calculates Precision@K for each test item
- Reports average Precision@5 across all test samples
- Uses room_type matching as relevance proxy

**Section 6: Cleanup & Results (Cell 15)**
- Displays final evaluation metrics
- Stops SparkSession to free resources
- Outputs summary statistics

### Getting Recommendations

#### Basic Usage

```python
# Get top 5 recommendations for a specific listing
recommendations = recommend_spark(
    item_name="Cozy Beach House in Galle",
    items_df=items_with_features,
    top_k=5
)

# Display recommendations with similarity scores
recommendations.select("name", "room_type", "similarity").show(truncate=False)
```

#### Advanced Usage

```python
# Get top 10 recommendations with all attributes
recommendations = recommend_spark(
    item_name="Your Listing Name",
    items_df=items_with_features,
    top_k=10
)

# Filter by minimum similarity threshold
high_similarity_recs = recommendations.filter(
    recommendations.similarity >= 0.8
)

# Sort by additional criteria
sorted_recs = recommendations.orderBy(
    ["similarity", "star_rating"], 
    ascending=[False, False]
)

# Export recommendations
recommendations.toPandas().to_csv("recommendations.csv", index=False)
```

#### Example Queries

```python
# Recommend similar private rooms
recommend_spark("Private Room in Colombo Fort", items_with_features, 5)

# Recommend similar entire homes
recommend_spark("Luxury Villa with Pool - Bentota", items_with_features, 10)

# Recommend similar budget accommodations
recommend_spark("Budget Hostel Ella", items_with_features, 5)
```

### Customizing Parameters

You can modify key parameters in the notebook:

```python
# Change number of recommendations
top_k = 10  # Default: 5

# Change evaluation sample size
sample_size = 50  # Default: 20

# Change normalization method
normalizer = Normalizer(inputCol="features", outputCol="norm_features", p=2.0)
# p=1.0 for L1 norm, p=2.0 for L2 norm

# Adjust Spark configurations for performance
spark = SparkSession.builder \
    .appName("Tourism Recommender") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()
```

## 📈 Model Architecture

### Detailed Pipeline Architecture

The recommendation system employs a sophisticated multi-stage pipeline built on Spark MLlib:

#### Stage 1: Data Ingestion & Validation

```
CSV File → Spark DataFrame → Schema Validation → Null Filtering
```

**Operations**:
- Reads CSV with automatic schema inference
- Validates data types and structure
- Removes records with missing critical fields
- Creates initial DataFrame with proper column types

**Output**: Clean DataFrame with validated records

#### Stage 2: Feature Selection & Preprocessing

```
Raw DataFrame → Column Selection → Data Type Conversion → Feature Subset
```

**Operations**:
- Extracts relevant columns (name, room_type, star_rating, number_of_guests)
- Casts numerical fields to appropriate types
- Creates unified schema for downstream processing

**Output**: Preprocessed DataFrame ready for feature engineering

#### Stage 3: Feature Engineering Pipeline

The ML pipeline consists of 4 sequential transformers:

**3.1 StringIndexer**
```python
StringIndexer(inputCol="room_type", outputCol="room_type_index")
```
- Converts categorical room types to numerical indices
- Example: "Private room" → 0, "Entire home/apt" → 1
- Handles unseen categories with "keep" strategy
- Creates label-to-index mapping for interpretability

**3.2 OneHotEncoder**
```python
OneHotEncoder(inputCol="room_type_index", outputCol="room_type_vec")
```
- Transforms indices into binary vector format
- Creates sparse vector representation for efficiency
- Example: Index 0 → [1.0, 0.0, 0.0], Index 1 → [0.0, 1.0, 0.0]
- Prevents ordinal relationship assumptions

**3.3 VectorAssembler**
```python
VectorAssembler(inputCols=["room_type_vec", "star_rating", "number_of_guests"], 
                outputCol="features")
```
- Combines all feature columns into single vector
- Creates dense or sparse vectors based on input
- Enables unified treatment of heterogeneous features
- Output: Feature vector of dimension N (varies by data)

**3.4 Normalizer**
```python
Normalizer(inputCol="features", outputCol="norm_features", p=2.0)
```
- Applies L2 (Euclidean) normalization
- Scales each feature vector to unit length
- Essential for accurate cosine similarity computation
- Formula: normalized_vector = vector / ||vector||₂

**Pipeline Execution**:
```
fit() → learns parameters from training data
transform() → applies learned transformations
```

#### Stage 4: Similarity Computation & Ranking

**Algorithm**: Content-Based Filtering with Cosine Similarity

```
Query Item → Extract Normalized Vector → Compute Dot Products → Rank by Score → Top-K Selection
```

**Mathematical Foundation**:

For two normalized vectors **v₁** and **v₂**:

```
Cosine Similarity = v₁ · v₂ = Σ(v₁ᵢ × v₂ᵢ)
```

Since vectors are L2-normalized (||v|| = 1):
```
Similarity Score = dot_product(norm_features₁, norm_features₂)
```

**Properties**:
- Range: [0, 1] for normalized vectors
- 1.0 = Identical items
- 0.0 = Completely dissimilar items
- Captures angular similarity in feature space

**Optimization**:
- Uses broadcast joins for efficient computation
- Leverages Spark's distributed computing
- Implements early filtering to reduce computation

#### Stage 5: Recommendation Delivery

```
Similarity Scores → Sorting → Top-K Selection → Result Formatting
```

**Operations**:
- Sorts all items by similarity score (descending)
- Excludes the query item itself
- Selects top K items (configurable)
- Returns DataFrame with item details and scores

#### Stage 6: Model Evaluation

**Metric**: Precision@K

```
For each test item:
    Generate K recommendations
    Count relevant items (same room_type)
    Calculate: Precision@K = relevant_count / K
    
Average Precision@K across all test items
```

**Evaluation Process**:
1. Sample random items from dataset (e.g., 20 items)
2. For each sampled item:
   - Generate top-K recommendations
   - Determine relevance (room_type matching)
   - Calculate Precision@K
3. Compute mean Precision@K across all samples
4. Report aggregate statistics

### Similarity Metric Deep Dive

#### Why Cosine Similarity?

**Advantages**:
- ✅ Magnitude-independent (focuses on direction)
- ✅ Effective for high-dimensional sparse data
- ✅ Computationally efficient with normalized vectors
- ✅ Intuitive interpretation (angular distance)
- ✅ Widely used in content-based filtering

**Computation Steps**:

1. **Vector Normalization**:
   ```
   For vector v: v_norm = v / sqrt(Σ vᵢ²)
   ```

2. **Dot Product Calculation**:
   ```
   similarity(A, B) = Σ(Aᵢ × Bᵢ) for normalized vectors
   ```

3. **Interpretation**:
   - 1.0: Vectors point in same direction (identical preferences)
   - 0.5: 60° angle between vectors
   - 0.0: Orthogonal vectors (no similarity)

#### Alternative Metrics (Not Used)

- **Euclidean Distance**: Sensitive to magnitude, less suitable for high dimensions
- **Manhattan Distance**: Less intuitive for feature similarity
- **Jaccard Similarity**: Better for set-based comparisons
- **Pearson Correlation**: Assumes linear relationships

### Architecture Scalability

The architecture is designed for horizontal scaling:

- **Data Parallelism**: Dataset partitioned across multiple nodes
- **Lazy Evaluation**: Optimizes execution plan before computation  
- **In-Memory Caching**: Stores frequently accessed data in RAM
- **Broadcast Variables**: Efficiently shares lookup tables across workers
- **Catalyst Optimizer**: Automatically optimizes query execution plans

## 📊 Evaluation Metrics

**Precision@K**: Measures the proportion of relevant items in the top-K recommendations

```
Precision@K = (Number of Relevant Items in Top-K) / K
```

**Relevance Proxy**: Items sharing the same `room_type` are considered relevant

**Evaluation Results**: The model is tested on a random sample of 20 items to compute average Precision@5

## 📁 Project Structure

```
BigData_Project/
├── A_Scalable_Content_Based_Recommendation_System_for_Sri_Lankan_Tourism.ipynb
├── sri_lanka_airbnb.csv
├── A-Scalable-Content-Based-Recommendation-System-for-Sri-Lankan-Tourism.pdf
├── BDA-Project-Demo.mkv
└── README.md
```

## 🔍 Key Components

### Section 1: Environment Setup
- Spark session initialization
- Library imports

### Section 2: Data Ingestion & Preprocessing
- CSV data loading
- Column selection and renaming
- Null value handling

### Section 3: Feature Engineering
- ML pipeline construction
- Categorical encoding
- Feature vector normalization

### Section 4: Recommendation Logic
- Cosine similarity calculation
- Top-K recommendation generation

### Section 5: Model Validation
- Precision@K evaluation
- Sample-based testing

### Section 6: Cleanup
- Resource deallocation

## 🎓 Use Cases

- **Tourism Platforms**: Suggest similar accommodations to users
- **Travel Planning**: Help tourists discover alternatives
- **Market Analysis**: Identify listing clusters and patterns
- **Price Optimization**: Understand competitive positioning

## ⚙️ Configuration

Key parameters you can adjust:

- `top_k`: Number of recommendations to generate (default: 5)
- `sample_size`: Number of items for evaluation (default: 20)
- `p`: Normalization parameter in Normalizer (default: 2.0 for L2 norm)

## 🚧 Future Enhancements

### Planned Features (Phase 2)

#### Enhanced Feature Engineering
- **Location-Based Features**:
  - Incorporate GPS coordinates (latitude, longitude)
  - Calculate geographical proximity between listings
  - Add region/city clustering for location-aware recommendations
  - Integrate popular tourist attraction distances
  
- **Price & Value Features**:
  - Price per night normalization
  - Price-to-rating ratio (value metric)
  - Seasonal price variations
  - Discount and special offer indicators
  
- **Amenity Analysis**:
  - Text mining of amenities lists
  - Binary features for popular amenities (WiFi, pool, parking, AC)
  - Amenity scoring based on frequency and importance
  - Luxury vs. budget amenity classification

- **Host Quality Metrics**:
  - Superhost status indicator
  - Host response rate and time
  - Number of listings per host
  - Host experience (years active)

#### Advanced Algorithms

- **Hybrid Recommendation System**:
  - Combine content-based with collaborative filtering
  - User-based recommendations if user interaction data available
  - Matrix factorization techniques (ALS, SVD)
  - Weighted ensemble of multiple algorithms
  
- **Deep Learning Integration**:
  - Neural collaborative filtering
  - Embedding layers for categorical features
  - Image similarity using CNN (if property images available)
  - Natural language processing for review sentiment

- **Context-Aware Recommendations**:
  - Seasonal adjustments (monsoon, peak season)
  - User context (family, couples, solo travelers)
  - Budget constraints integration
  - Trip duration considerations

#### Real-Time Systems

- **Streaming Recommendations**:
  - Spark Structured Streaming for real-time updates
  - Real-time model updates as new listings added
  - Live popularity metrics
  - Dynamic pricing integration

- **API Development**:
  - RESTful API using Flask or FastAPI
  - GraphQL endpoint for flexible queries
  - Authentication and rate limiting
  - Caching layer for frequently requested recommendations

#### User Experience & Feedback

- **Interactive Features**:
  - User preference learning
  - Explicit feedback collection (likes, dislikes)
  - Implicit feedback tracking (clicks, view duration)
  - A/B testing framework for algorithm improvements

- **Personalization**:
  - User profile creation
  - Historical preference tracking
  - Multi-session learning
  - Diverse recommendation strategies

#### Deployment & Production

- **Cloud Deployment**:
  - Containerization with Docker
  - Kubernetes orchestration for scaling
  - AWS/Azure/GCP cloud infrastructure
  - Load balancing and auto-scaling

- **Monitoring & Optimization**:
  - Performance monitoring dashboard
  - Model drift detection
  - Automatic retraining pipelines
  - Resource usage optimization
  - Cost tracking and optimization

#### Enhanced Evaluation

- **Advanced Metrics**:
  - Mean Average Precision (MAP)
  - Normalized Discounted Cumulative Gain (NDCG)
  - Diversity metrics (Intra-List Similarity)
  - Coverage analysis
  - Novelty and serendipity metrics

- **Online Evaluation**:
  - Click-through rate (CTR) tracking
  - Conversion rate monitoring
  - User satisfaction surveys
  - Long-term engagement metrics

### Research Directions

- Multi-modal learning with text, images, and structured data
- Explainable AI for transparent recommendations
- Fairness and bias detection in recommendations
- Cross-domain recommendations (hotels, activities, restaurants)
- Temporal pattern analysis in booking behavior

### Timeline

- **Q2 2025**: Enhanced features and hybrid algorithms
- **Q3 2025**: Real-time systems and API development
- **Q4 2025**: Cloud deployment and production monitoring
- **2026+**: Advanced ML techniques and research contributions

## 📝 Notes

- The current implementation uses `room_type` as the primary feature for demonstration
- Feature engineering pipeline is designed to be easily extensible
- Evaluation uses a proxy for relevance due to lack of explicit user feedback
- Spark resources are properly released at the end of execution

## � Author & Ownership

**Nipuna Lakruwan**

- **Role**: Sole Owner, Developer, and Maintainer
- **Copyright**: © 2025 Nipuna Lakruwan. All Rights Reserved.
- **Project**: A Scalable Content-Based Recommendation System for Sri Lankan Tourism

This project was conceptualized, designed, developed, and documented entirely by Nipuna Lakruwan as part of a Big Data Analytics initiative focusing on the Sri Lankan tourism sector.

### Ownership Statement

This project, including all source code, documentation, datasets (where applicable), and associated materials, is the exclusive intellectual property of **Nipuna Lakruwan**. No other individuals or organizations have contributed to or hold rights to this work.

### Permissions

- **Personal Use**: Allowed for learning and reference
- **Commercial Use**: Requires explicit written permission from the owner
- **Modification**: Allowed with proper attribution to the original author
- **Distribution**: Requires permission and must retain authorship information

## 📄 License

**Copyright © 2025 Nipuna Lakruwan. All Rights Reserved.**

This project is provided for **educational and research purposes only**. 

### Terms of Use

- This software is provided "as is" without warranty of any kind
- The author (Nipuna Lakruwan) retains all rights to this work
- Usage for academic study and learning is permitted with proper attribution
- Commercial usage requires explicit written consent from the author
- Redistribution must include this copyright notice and attribution
- Any derivative works must acknowledge the original author

### Attribution Requirements

If you reference, use, or build upon this work, please provide clear attribution:

```
Original Work: A Scalable Content-Based Recommendation System for Sri Lankan Tourism
Author: Nipuna Lakruwan
Year: 2025
```

## 📞 Contact

### Get in Touch

For questions, suggestions, collaborations, or licensing inquiries regarding this project, please contact:

**Nipuna Lakruwan**
- **Project Owner & Lead Developer**

### Communication Guidelines

- **Technical Questions**: Feel free to ask about implementation details, algorithms, or architecture
- **Bug Reports**: Report any issues you encounter with detailed reproduction steps
- **Feature Suggestions**: Propose enhancements or additional features
- **Collaboration Inquiries**: Discuss potential partnerships or extensions
- **Commercial Licensing**: Request permissions for commercial use
- **Academic Citations**: Request proper citation format for academic work

### Response Time

I aim to respond to all inquiries within 48-72 hours. For urgent matters, please indicate priority in your message subject line.

---

**Note**: Ensure you have sufficient memory allocated to Spark when working with larger datasets. Adjust Spark configuration parameters as needed for your environment.
