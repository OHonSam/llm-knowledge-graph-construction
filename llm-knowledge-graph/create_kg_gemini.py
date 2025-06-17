import os
from loguru import logger
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

logger.add("create_kg.log", rotation="10 MB", level="DEBUG")

from dotenv import load_dotenv
load_dotenv()

# DOCS_PATH = "llm-knowledge-graph/data/course/pdfs"
DOCS_PATH = "llm-knowledge-graph/data/custom_pdfs"

try:
    llm = ChatGoogleGenerativeAI(
        google_api_key=os.getenv('GEMINI_API_KEY'),
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    logger.success("LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    raise

try:
    embedding_provider = GoogleGenerativeAIEmbeddings(
        google_api_key=os.getenv('GEMINI_API_KEY'),
        model="models/gemini-embedding-exp-03-07"
        )
    logger.success("Embedding provider initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize embedding provider: {e}")
    raise

try:
    graph = Neo4jGraph(
        url=os.getenv('NEO4J_URI'),
        username=os.getenv('NEO4J_USERNAME'),
        password=os.getenv('NEO4J_PASSWORD')
    )
    logger.success("Neo4j graph connection established")
except Exception as e:
    logger.error(f"Failed to connect to Neo4j: {e}")
    raise

try:
    custom_system_template = r"""
    Bạn là một chuyên gia về Y học cổ truyền Việt Nam và Trung Quốc. Nhiệm vụ của bạn là trích xuất các thực thể và mối quan hệ từ văn bản y học cổ truyền một cách chính xác và có hệ thống.

    HƯỚNG DẪN TRÍCH XUẤT THỰC THỂ:

    1. **Thảo dược**: Các loại thuốc nam đơn lẻ
    - Ví dụ: nhân sâm, cam thảo, đương quy, hoàng kỳ, bạch truật
    - Bao gồm cả tên Hán Việt và tên phổ thông

    2. **Bài thuốc**: Các công thức thuốc phức hợp
    - Ví dụ: Lục vị hoàn, Bát trân thang, Tứ quân tử thang
    - Các phương thuốc cổ truyền và hiện đại

    3. **Triệu chứng**: Các biểu hiện bệnh lý cụ thể
    - Ví dụ: sốt, ho, đau đầu, buồn nôn, tiêu chảy, mất ngủ
    - Các triệu chứng theo thuật ngữ y học cổ truyền

    4. **Bệnh lý**: Tên các bệnh và hội chứng
    - Ví dụ: cảm mạo, trúng phong, tiêu chảy, hen suyễn
    - Bao gồm cả tên bệnh hiện đại và cổ truyền

    5. **Tạng phủ**: Các tạng phủ theo lý thuyết y học cổ truyền
    - Năm tạng: Tâm, Can, Tỳ, Phế, Thận
    - Sáu phủ: Đởm, Vị, Tiểu trường, Đại trường, Tam tiêu, Bàng quang

    6. **Kinh lạc**: Hệ thống kinh mạch trong y học cổ truyền
    - 12 kinh chính, 8 kỳ kinh
    - Các đường dẫn khí huyết

    7. **Chứng bệnh**: Các chứng theo y học cổ truyền
    - Ví dụ: hư hỏa, thực hàn, phong hàn, thấp nhiệt
    - Phân biệt hư/thực, hàn/nhiệt, âm/dương

    8. **Chẩn đoán**: Phương pháp và kết quả chẩn đoán
    - Tứ chẩn: vọng, văn, vấn, thiết
    - Biện chứng luận trị

    9. **Phương pháp trị liệu**: Các phương pháp điều trị
    - Châm cứu, bấm huyệt, xoa bóp
    - Phương pháp dùng thuốc

    10. **Huyệt đạo**: Các huyệt vị trong châm cứu
        - Tên huyệt và vị trí
        - Công dụng của từng huyệt

    11. **Nhân vật**: Các danh y, tác giả
        - Y gia nổi tiếng trong lịch sử
        - Người sáng tạo các bài thuốc

    12. **Khái niệm**: Các lý thuyết và nguyên lý y học cổ truyền
        - Âm dương, ngũ hành, khí huyết
        - Các khái niệm triết học y học

    13. **Thực phẩm**: Các loại thực phẩm có tác dụng chữa bệnh
        - Thực phẩm bổ dưỡng, thực phẩm kiêng kỵ
        - Dinh dưỡng trị liệu

    14. **Phương pháp bào chế**: Cách chế biến thuốc
        - Sao, nướng, tẩm, ngâm
        - Các kỹ thuật chế biến truyền thống

    15. **Vùng miền**: Địa danh, vùng trồng thuốc
        - Nơi sản xuất thuốc chất lượng
        - Vùng địa lý có ý nghĩa y học

    HƯỚNG DẪN TRÍCH XUẤT MỐI QUAN HỆ:

    1. **ĐIỀU_TRỊ**: Thảo dược/Bài thuốc → Bệnh lý/Triệu chứng/Chứng bệnh
    - "Cam thảo ĐIỀU_TRỊ ho"
    - "Lục vị hoàn ĐIỀU_TRỊ thận hư"

    2. **CHỨA**: Bài thuốc → Thảo dược
    - "Lục vị hoàn CHỨA thục địa"
    - "Tứ quân tử thang CHỨA nhân sâm"

    3. **TÁC_ĐỘNG**: Thảo dược/Bài thuốc → Tạng phủ/Kinh lạc
    - "Nhân sâm TÁC_ĐỘNG Tâm kinh"
    - "Đương quy TÁC_ĐỘNG Can tạng"

    4. **CÂN_BẰNG**: Các yếu tố điều hòa lẫn nhau
    - "Âm CÂN_BẰNG Dương"
    - "Khí CÂN_BẰNG Huyết"

    5. **THUỘC_VỀ**: Quan hệ phân loại
    - "Đởm THUỘC_VỀ Phủ"
    - "Tâm THUỘC_VỀ Tạng"

    6. **KÍCH_THÍCH**: Tác động kích hoạt
    - "Châm cứu KÍCH_THÍCH huyệt đạo"
    - "Thuốc nóng KÍCH_THÍCH dương khí"

    7. **NẰM_Ở**: Vị trí địa lý
    - "Huyệt NẰM_Ở vị trí cụ thể"
    - "Thuốc NẰM_Ở vùng miền"

    8. **PHÁT_TRIỂN_TỪ**: Nguồn gốc, xuất xứ
        - "Bài thuốc PHÁT_TRIỂN_TỪ lý thuyết cổ"
        - "Phương pháp PHÁT_TRIỂN_TỪ kinh nghiệm"

    9. **SỬ_DỤNG_TRONG**: Ứng dụng trong hoàn cảnh
        - "Thảo dược SỬ_DỤNG_TRONG bài thuốc"
        - "Phương pháp SỬ_DỤNG_TRONG điều trị"

    10. **CHẾ_BIẾN_THÀNH**: Quá trình chế biến
        - "Thảo dược tươi CHẾ_BIẾN_THÀNH thuốc khô"
        - "Nguyên liệu CHẾ_BIẾN_THÀNH thành phẩm"

    11. **SÁNG_TẠO_BỞI**: Tác giả, người tạo ra
        - "Bài thuốc SÁNG_TẠO_BỞI danh y"
        - "Phương pháp SÁNG_TẠO_BỞI nhân vật"

    12. **CHẨN_ĐOÁN_BỞI**: Phương pháp chẩn đoán
        - "Bệnh lý CHẨN_ĐOÁN_BỞI phương pháp"
        - "Chứng bệnh CHẨN_ĐOÁN_BỞI tứ chẩn"

    13. **TRỒNG_TẠI**: Nơi canh tác, sản xuất
        - "Thảo dược TRỒNG_TẠI vùng miền"
        - "Nguyên liệu TRỒNG_TẠI địa phương"

    QUY TẮC QUAN TRỌNG:
    - CHỈ trích xuất thông tin có trong văn bản, không suy đoán
    - Sử dụng tên tiếng Việt chuẩn cho thực thể
    - Ưu tiên thuật ngữ y học cổ truyền chính thống
    - Phân biệt rõ ràng giữa các loại thực thể
    - Xác định mối quan hệ dựa trên ngữ cảnh cụ thể
    - Tránh tạo ra các mối quan hệ mơ hồ hoặc không chắc chắn

    Trả về kết quả dưới dạng JSON với cấu trúc nodes và relationships được định nghĩa rõ ràng theo đúng định dạng yêu cầu.
    """

    custom_human_template = r"""
    Phân tích văn bản y học cổ truyền sau và trích xuất entities và relationships:
    
    Văn bản: {input}
    
    Hãy trả về kết quả dưới dạng JSON với cấu trúc nodes và relationships.
    """
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", custom_system_template),
        ("human", custom_human_template)
    ])

    doc_transformer = LLMGraphTransformer(
        llm=llm,
        prompt=chat_prompt,
        allowed_nodes = [
            "Thảo dược",             # e.g., Nhân sâm, Cam thảo
            "Bài thuốc",             # e.g., Lục vị hoàn, Bát trân thang
            "Triệu chứng",           # e.g., Sốt, Ho, Mất ngủ
            "Bệnh lý",               # e.g., Cảm mạo, Trúng phong
            "Tạng phủ",              # e.g., Tâm, Can, Tỳ, Thận, Phế
            "Kinh lạc",              # e.g., Kinh Phế, Kinh Tỳ
            "Chứng bệnh",            # e.g., Hư hỏa, Phong hàn
            "Chẩn đoán",             # e.g., Vọng, Văn, Vấn, Thiết
            "Phương pháp trị liệu",  # e.g., Châm cứu, Cạo gió, Xoa bóp
            "Huyệt đạo",             # e.g., Huyệt Túc Tam Lý (ST36), Hợp Cốc (LI4)
            "Nhân vật",              # e.g., Hải Thượng Lãn Ông, Tuệ Tĩnh
            "Khái niệm",             # e.g., Âm-Dương, Ngũ Hành, Khí, Huyết
            "Thực phẩm",             # e.g., Gừng, Tỏi (Dưỡng sinh qua ẩm thực)
            "Phương pháp bào chế",   # e.g., Sắc, Tán, Hoàn, Cao
            "Vùng miền"              # e.g., Yên Bái, Hà Nam (nơi mọc cây thuốc)
        ],
        # allowed_relationships = [
        #     "ĐIỀU_TRỊ",          # Medicine/Formula → Disease/Symptom (strong medical connection)
        #     "CHỨA",              # Formula → Herb (clear composition)
        #     "TÁC_ĐỘNG",          # Medicine → Organ (direct physiological effect)
        #     "THUỘC_VỀ",          # Classification (Organ → Type, Herb → Category)
        #     "SỬ_DỤNG",           # Method → Context (specific usage)
        #     "SÁNG_TẠO_BỞI",      # Formula → Person (clear authorship)
        #     "TRỒNG_TẠI",         # Herb → Region (geographical origin)
        #     "CHẨN_ĐOÁN_BỞI",     # Disease → Method (diagnostic relationship)
        #     "CÂN_BẰNG",
        #     "KÍCH_THÍCH",
        #     "CHẾ_BIẾN_THÀNH",
        #     "NẰM_Ở",
        #     "PHÁT_TRIỂN_TỪ",
        # ]
        )
    
    logger.success("Document transformer initialized")
except Exception as e:
    logger.error(f"Failed to initialize document transformer: {e}")
    raise

# Load and split the documents
logger.info("Loading documents...")
try:
    loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} documents")
except Exception as e:
    logger.error(f"Failed to load documents: {e}")
    raise

logger.info("Splitting documents into chunks...")
try:
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1500,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Created {len(chunks)} chunks from documents")
except Exception as e:
    logger.error(f"Failed to split documents: {e}")
    raise

for chunk in chunks:

    filename = os.path.basename(chunk.metadata["source"])
    chunk_id = f'{filename}.{chunk.metadata["page"]}'
    print("Processing -", chunk_id)

    # Embed the chunk
    chunk_embedding = embedding_provider.embed_query(chunk.page_content)

    # Add the Document and Chunk nodes to the graph
    properties = {
        "filename": filename,
        "chunk_id": chunk_id,
        "text": chunk.page_content,
        "embedding": chunk_embedding
    }
    
    graph.query("""
        MERGE (d:Document {id: $filename})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text
        MERGE (d)<-[:PART_OF]-(c)
        WITH c
        CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
        """, 
        properties
    )

    # Generate the entities and relationships from the chunk
    graph_docs = doc_transformer.convert_to_graph_documents([chunk])

    # Map the entities in the graph documents to the chunk node
    for graph_doc in graph_docs:
        chunk_node = Node(
            id=chunk_id,
            type="Văn Bản"
        )

        for node in graph_doc.nodes:

            graph_doc.relationships.append(
                Relationship(
                    source=chunk_node,
                    target=node, 
                    type="Có_Thực_Thể"
                    )
                )

    # add the graph documents to the graph
    graph.add_graph_documents(graph_docs)

# Create the vector index
graph.query("""
    CREATE VECTOR INDEX `chunkVector`
    IF NOT EXISTS
    FOR (c: Chunk) ON (c.textEmbedding)
    OPTIONS {indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
    }};""")