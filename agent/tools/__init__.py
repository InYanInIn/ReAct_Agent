from agent.tools.analyzer import CodeAnalyzer
from agent.tools.evaluator import CodeEvaluator
from agent.tools.fixer import CodeFixer
from agent.tools.sandbox import SandboxExecutor


def get_tools():
    return {
        "analyzer": CodeAnalyzer.analyze_code,
        "code_fixer": CodeFixer.generate_fix,
        "evaluator": CodeEvaluator.evaluate_fix,
        "sandbox": SandboxExecutor.execute_code
    }


__all__ = ['get_tools', 'CodeAnalyzer', 'CodeFixer', 'CodeEvaluator', 'SandboxExecutor']
