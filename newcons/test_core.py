import os
import sys
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.vector_store import build_hybrid_knowledge_base
from engine.rag_pipeline import get_answer_complex, _init_db
from algorithms.linucb import linucb_agent
import sqlite3

class TestEngine(unittest.TestCase):
    def test_linucb_select_arm(self):
        query_vec = [0.1] * 12  # Dummy embedding
        arm_idx, final_alpha, context_vec = linucb_agent.select_arm(query_vec)
        self.assertTrue(0 <= arm_idx < linucb_agent.n_arms)
        self.assertEqual(len(context_vec), 8) # feature_dim is 8
        self.assertEqual(context_vec[0], 0.1)
        
        linucb_agent.update(arm_idx, context_vec, 1.0)
        
    def test_build_knowledge_base(self):
        # Create a dummy txt file
        with open("dummy_test.txt", "w", encoding="utf-8") as f:
            f.write("这是一个关于原神的测试文档，里面包含了黄泉、星穹铁道等内容。")
        
        vs, splits, count = build_hybrid_knowledge_base("dummy_test.txt")
        self.assertIsNotNone(vs)
        self.assertTrue(len(splits) > 0)
        self.assertTrue(count > 0)
        
        # Check that chroma_db_data exists
        self.assertTrue(os.path.exists("./chroma_db_data"))
        
        os.remove("dummy_test.txt")

    @patch('engine.rag_pipeline.ChatGoogleGenerativeAI')
    @patch('engine.rag_pipeline.analyze_user_query')
    def test_rag_pipeline_negative_feedback(self, mock_analyze, mock_llm):
        mock_analyze.return_value = ("negative", [("黄泉", "ENTITY")], ["强度党"])
        
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = MagicMock(content="Test answer")
        mock_llm.return_value = mock_llm_instance
        
        # Setup dummy DB
        _init_db()
        
        res = get_answer_complex(
            vectorstore=None,  # skip retrieval for basic test
            bm25_retriever=None,
            question="这个游戏太烂了",
            model_type="cloud",
            use_emotion=True,
            use_auto_alpha=True
        )
        
        self.assertEqual(res["emotion"], "negative")
        self.assertIn("强度党", res["persona"])
        self.assertTrue(res["arm_idx"] >= 0)
        self.assertEqual(len(res["context_vec"]), 8)
        
        # Check SQLite DB
        conn = sqlite3.connect("community_feedback_log.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM feedback_logs ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(row)
        self.assertEqual(row[2], "这个游戏太烂了")
        self.assertEqual(row[3], "negative")
        self.assertEqual(row[4], "强度党")

if __name__ == '__main__':
    unittest.main()
