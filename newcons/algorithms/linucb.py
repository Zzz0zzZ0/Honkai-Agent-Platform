import numpy as np
import re

class LinUCBEngine:
    def __init__(self, alpha=1.0, n_arms=5, feature_dim=3):
        self.alpha = alpha
        self.n_arms = n_arms
        self.feature_dim = feature_dim
        self.arm_values =[0.0, 0.25, 0.5, 0.75, 1.0]
        self.Aa = {} 
        self.ba = {} 
        for i in range(self.n_arms):
            self.Aa[i] = np.identity(self.feature_dim)
            self.ba[i] = np.zeros(self.feature_dim)

    def extract_context_features(self, query):
        length_norm = min(len(query) / 20.0, 1.0)
        # 修复：正确匹配特殊字符（非字母数字和空格）
        special_chars = len(re.findall(r'[^A-Za-z0-9_\s]', query))
        density = special_chars / (len(query) + 1.0)
        return np.array([1.0, length_norm, density])

    def select_arm(self, query):
        x_t = self.extract_context_features(query)
        p_t = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            A_inv = np.linalg.inv(self.Aa[i])
            theta_hat = A_inv.dot(self.ba[i])
            uncertainty = self.alpha * np.sqrt(x_t.dot(A_inv).dot(x_t))
            p_t[i] = theta_hat.dot(x_t) + uncertainty
        best_idx = np.argmax(p_t)
        return best_idx, self.arm_values[best_idx], x_t

    def update(self, arm_idx, x_t, reward):
        self.Aa[arm_idx] += np.outer(x_t, x_t)
        self.ba[arm_idx] += reward * x_t

# 全局实例化 LinUCB 智能体
linucb_agent = LinUCBEngine(alpha=0.5)