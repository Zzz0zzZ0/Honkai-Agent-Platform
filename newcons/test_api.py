import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.server import app, global_memory

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Reset memory
        global_memory["vectorstore"] = None
        global_memory["bm25"] = None
        global_memory["all_splits"] = []

    def test_upload_memory(self):
        file_content = "这是一段针对 API 测试的文档，提到了原神和崩坏星穹铁道。".encode("utf-8")
        response = self.client.post(
            "/upload_memory",
            files={"file": ("test_doc.txt", file_content, "text/plain")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(len(global_memory["all_splits"]) > 0)
        self.assertIsNotNone(global_memory["bm25"])
        
    @patch('api.server.get_answer_complex')
    def test_chat_without_agent(self, mock_get_answer):
        mock_get_answer.return_value = {
            "answer": "原神和崩坏星穹铁道",
            "persona": ["测试党"],
            "arm_idx": 1,
            "context_vec": [0.1] * 8
        }
        chat_req = {
            "query": "文档里提到了什么游戏？",
            "use_agent": False,
            "model_type": "cloud",
            "use_emotion": True,
            "use_auto_alpha": True,
            "alpha": 0.5,
            "k_param": 3,
            "temp_param": 0.1
        }
        
        response = self.client.post("/chat", json=chat_req)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["answer"], "原神和崩坏星穹铁道")
        self.assertEqual(data["persona"], ["测试党"])
        self.assertEqual(data["arm_idx"], 1)
        self.assertEqual(len(data["context_vec"]), 8)

    @patch('algorithms.linucb.LinUCBEngine.update')
    def test_feedback(self, mock_update):
        feedback_req = {
            "arm_idx": 2,
            "context_vec": [0.5] * 8,
            "reward": 1.0
        }
        response = self.client.post("/feedback", json=feedback_req)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        mock_update.assert_called_once_with(2, [0.5] * 8, 1.0)

if __name__ == '__main__':
    unittest.main()
