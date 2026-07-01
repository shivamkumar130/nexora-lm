@echo off
cd /d D:\Nexora-AI

echo ==========================================
echo DATASET
echo ==========================================
python -m training.data.build_training_corpus
python -m training.data.dataset

echo ==========================================
echo KNOWLEDGE INDEX
echo ==========================================
python knowledge\faiss\build_index.py

echo ==========================================
echo TOKENIZER
echo ==========================================
python -m tokenizer.train_tokenizer
python -m tokenizer.tokenizer

echo ==========================================
echo EMBEDDINGS
echo ==========================================
python -m model.embedding.token_embedding
python -m model.embedding.position_embedding

echo ==========================================
echo ATTENTION
echo ==========================================
python -m model.attention.self_attention
python -m model.attention.multihead_attention

echo ==========================================
echo TRANSFORMER
echo ==========================================
python -m model.transformer.transformer_block
python -m model.transformer.nexoralm

echo ==========================================
echo TRAINING COMPONENTS
echo ==========================================
python -m training.loss.cross_entropy
python -m training.optimizer.adamw
python -m training.scheduler.cosine_scheduler

echo ==========================================
echo TRAINING
echo ==========================================
python -m training.trainer.trainer
python -m training.train

echo ==========================================
echo EVALUATION
echo ==========================================
python -m training.evaluate

echo ==========================================
echo INFERENCE
echo ==========================================
python -m inference.engine.load_model

echo ==========================================
echo INTEGRATION TEST
echo ==========================================
python -m tests.integration.system_test

echo ==========================================
echo ALL TASKS COMPLETED
echo ==========================================

pause