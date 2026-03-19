import re
from collections import defaultdict

class SimpleBPE:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
    
    def get_word_freqs(self, text):
        """统计单词频率"""
        # 简单分词：按空格和标点分割
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        freqs = defaultdict(int)
        for word in words:
            # BPE用</w>标记词尾
            freqs[' '.join(list(word)) + ' </w>'] += 1
        return freqs
    
    def get_stats(self, word_freqs):
        """统计相邻token对"""
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
    
    def merge_vocab(self, pair, word_freqs):
        """合并指定的token对"""
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        new_word_freqs = {}
        for word in word_freqs:
            new_word = pattern.sub(''.join(pair), word)
            new_word_freqs[new_word] = word_freqs[word]
        
        return new_word_freqs
    
    def train(self, text):
        """训练BPE"""
        print("开始训练BPE...")
        
        # 初始化
        word_freqs = self.get_word_freqs(text)
        print(f"初始词数: {len(word_freqs)}")
        
        # 构建初始词表（所有字符）
        self.vocab = set()
        for word in word_freqs:
            self.vocab.update(word.split())
        
        # 迭代合并
        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            pairs = self.get_stats(word_freqs)
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            word_freqs = self.merge_vocab(best_pair, word_freqs)
            self.merges.append(best_pair)
            self.vocab.add(''.join(best_pair))
            
            if (i + 1) % 100 == 0:
                print(f"已合并 {i+1}/{num_merges} 对，词表大小: {len(self.vocab)}")
        
        print(f"训练完成，最终词表大小: {len(self.vocab)}")
        print(f"词表: {self.vocab}")
    
    def tokenize(self, text):
        """分词"""
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        tokens = []
        
        for word in words:
            word = ' '.join(list(word)) + ' </w>'
            
            # 应用所有合并规则
            for merge in self.merges:
                bigram = re.escape(' '.join(merge))
                pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
                word = pattern.sub(''.join(merge), word)
            
            tokens.extend(word.split())
        
        return tokens
    
class AdvancedBPE(SimpleBPE):
    def __init__(self, vocab_size=10000):
        super().__init__(vocab_size)
        # 特殊token
        self.special_tokens = {
            '<pad>': 0,      # 填充
            '<unk>': 1,      # 未知词
            '<bos>': 2,      # 句子开始
            '<eos>': 3,      # 句子结束
            '<mask>': 4,     # 掩码（BERT用）
        }
    
    def encode(self, text, add_special_tokens=True):
        """文本 → token ids"""
        tokens = self.tokenize(text)
        
        # 转id
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.special_tokens['<unk>'])
        
        # 添加特殊token
        if add_special_tokens:
            ids = [self.special_tokens['<bos>']] + ids + [self.special_tokens['<eos>']]
        
        return ids
    
    def decode(self, ids, skip_special_tokens=True):
        """token ids → 文本"""
        id_to_token = {v: k for k, v in self.vocab.items()}
        
        tokens = []
        for id in ids:
            if id in id_to_token:
                token = id_to_token[id]
                # 跳过特殊token
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
        
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')  # 词尾标记转空格
        return text.strip()

# 使用示例
text = """
自然语言处理是人工智能的重要方向。
大语言模型如GPT、BERT等在NLP任务中表现出色。
Tokenizer是连接文本和模型的桥梁。
"""

tokenizer = SimpleBPE(vocab_size=80)
tokenizer.train(text)

# 测试
test_text = "大语言模型"
tokens = tokenizer.tokenize(test_text)
print(f"\n测试: {test_text}")
print(f"分词结果: {tokens}")