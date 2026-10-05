from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base

class SourceDocuments(Base):
    __tablename__ = "source_documents"
    id = Column(String, primary_key=True, index=True)
    file_name = Column(String, index=True)
    grade = Column(Integer, nullable=True)
    subject = Column(String, nullable=True)
    
    chunks = relationship("ContentChunks", back_populates="document")

class ContentChunks(Base):
    __tablename__ = "content_chunks"
    id = Column(String, primary_key=True, index=True)
    source_id = Column(String, ForeignKey("source_documents.id"))
    chunk_text = Column(Text)
    topic = Column(String, nullable=True)
    subtopic = Column(String, nullable=True)
    page_number = Column(Integer, nullable=True)
    slide_number = Column(Integer, nullable=True)
    video_timestamp = Column(String, nullable=True)

    document = relationship("SourceDocuments", back_populates="chunks")
    questions = relationship("Questions", back_populates="chunk")

class Questions(Base):
    __tablename__ = "questions"
    id = Column(String, primary_key=True, index=True)
    source_chunk_id = Column(String, ForeignKey("content_chunks.id"))
    question = Column(Text)
    type = Column(String) # MCQ, True/False, Fill in the blank, Numerical, Short Answer
    options = Column(Text) # JSON string of array
    answer = Column(String)
    difficulty = Column(String) # easy, medium, hard
    page_number = Column(Integer, nullable=True)
    slide_number = Column(Integer, nullable=True)
    video_timestamp = Column(String, nullable=True)
    
    chunk = relationship("ContentChunks", back_populates="questions")

class StudentAnswers(Base):
    __tablename__ = "student_answers"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    student_id = Column(String, index=True)
    question_id = Column(String, ForeignKey("questions.id"))
    selected_answer = Column(String)
    is_correct = Column(Boolean)
