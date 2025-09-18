#!/usr/bin/env python3
import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Set, Tuple
import requests
from huggingface_hub import (
    hf_hub_download, 
    login, 
    model_info,
    list_repo_files,
    HfApi,
    repo_exists
)
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelInspector:
    """Enhanced HuggingFace model inspector with comprehensive analysis."""
    
    QUANT_PRECISION_MAPPING = {
        'int4': 'INT4',
        'int8': 'INT8', 
        'fp16': 'FP16',
        'bf16': 'BF16',
        'mxfp4': 'MXFP4',
        '4bit': '4-bit',
        '4-bit': '4-bit',
        '8bit': '8-bit',
        '8-bit': '8-bit'
    }
    
    QUANT_METHOD_INDICATORS = {
        'bnb': 'BitsAndBytes',
        'gptq': 'GPTQ', 
        'awq': 'AWQ',
        'ggml': 'GGML/GGUF',
        'gguf': 'GGML/GGUF'
    }
    
    ARCHITECTURE_CONFIGS = {
        'transformer_modern': {
            'types': ['llama', 'mistral', 'qwen', 'phi'],
            'params': ['intermediate_size', 'num_attention_heads'],
            'multiplier': lambda h, l, i, v: v * h * 2 + l * (h * h * 4 + h * i * 2 + h * 2)
        },
        'transformer_classic': {
            'types': ['gpt2', 'gpt_neox'],
            'params': [],
            'multiplier': lambda h, l, i, v: v * h * 2 + l * (h * h * 12 + h * 2)
        }
    }
    
    def __init__(self, token: Optional[str] = None):
        """Initialize the model inspector."""
        self.api = HfApi()
        self.token = token
        if token:
            logger.info("🔐 Authenticating with HuggingFace token")
            login(token=token, add_to_git_credential=False)

    def _safe_api_request(self, repo_id: str) -> Optional[Dict]:
        """Safely request model info from HF API with comprehensive error handling."""
        try:
            info = self.api.model_info(repo_id, token=self.token)
            return {
                attr: getattr(info, attr, default)
                for attr, default in [
                    ('downloads', 0), ('likes', 0), ('tags', []),
                    ('pipeline_tag', None), ('library_name', None),
                    ('created_at', None), ('last_modified', None),
                    ('private', False), ('gated', False), ('disabled', False),
                    ('author', None), ('sha', None)
                ]
            } | {'siblings': [s.rfilename for s in getattr(info, 'siblings', [])]}
        except Exception as e:
            logger.warning(f"Could not fetch model info from API: {e}")
            return None

    def _load_json_file(self, repo_id: str, filename: str) -> Optional[Dict]:
        """Download and load a JSON file from a Hugging Face repo."""
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename, token=self.token)
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load {filename}: {e}")
            return None

    def _estimate_parameters(self, config: Dict) -> Optional[int]:
        """Enhanced parameter estimation for various model architectures."""
        try:
            model_type = config.get("model_type", "").lower()
            required_params = {
                'hidden_size': config.get("hidden_size") or config.get("d_model"),
                'num_layers': config.get("num_hidden_layers") or config.get("n_layers") or config.get("num_layers"),
                'vocab_size': config.get("vocab_size")
            }
            
            if not all(required_params.values()):
                return None
            
            h, l, v = required_params.values()
            
            # Find matching architecture
            for arch_name, arch_config in self.ARCHITECTURE_CONFIGS.items():
                if model_type in arch_config['types']:
                    intermediate_size = config.get("intermediate_size", h * 4)
                    return int(arch_config['multiplier'](h, l, intermediate_size, v))
            
            # Generic estimation fallback
            return int(v * h * 2 + l * h * h * 12)
            
        except Exception as e:
            logger.warning(f"Parameter estimation failed: {e}")
            return None

    def _detect_quantization(self, config: Dict) -> Tuple[bool, Set[str]]:
        """Detect quantization types with improved logic."""
        detected_types = set()
        
        # Combine all searchable text
        search_targets = [
            (str(config), "config"),
            (config.get('model_type', ''), "model_type")
        ]
        
        for text, source in search_targets:
            text_lower = text.lower()
            
            # Check precision types
            for indicator, quant_type in self.QUANT_PRECISION_MAPPING.items():
                if indicator in text_lower:
                    detected_types.add(quant_type)
            
            # Check quantization methods
            for indicator, method in self.QUANT_METHOD_INDICATORS.items():
                if indicator in text_lower:
                    detected_types.add(method)
            
            # Check for generic quantization indicators
            generic_indicators = ['quant', 'bits']
            for indicator in generic_indicators:
                if indicator in text_lower and not detected_types:
                    detected_types.add('Quantized' if indicator == 'quant' else 'Multi-bit')
        
        # Special case: quantization_config presence
        if config.get('quantization_config'):
            detected_types.add('BitsAndBytes')
        
        return len(detected_types) > 0, detected_types

    def _analyze_files(self, siblings: List[str]) -> Dict[str, Any]:
        """Analyze repository files to determine formats and characteristics."""
        file_mappings = {
            'safetensors': ('.safetensors', 'SafeTensors'),
            'pytorch': (('.bin', '.pt', '.pth'), 'PyTorch'), 
            'gguf': ('.gguf', 'GGUF'),
            'onnx': ('.onnx', 'ONNX')
        }
        
        analysis = {
            'total_files': len(siblings),
            'formats': set(),
            'model_files': [],
            'config_files': [],
            'other_files': []
        }
        
        # Add individual format flags
        for format_key in file_mappings:
            analysis[f'has_{format_key}'] = False
        
        config_file_names = {
            'config.json', 'tokenizer.json', 'tokenizer_config.json',
            'generation_config.json', 'model.safetensors.index.json'
        }
        
        for file in siblings:
            file_lower = file.lower()
            categorized = False
            
            # Check each format type
            for format_key, (extensions, format_name) in file_mappings.items():
                extensions = extensions if isinstance(extensions, tuple) else (extensions,)
                if any(file_lower.endswith(ext) for ext in extensions):
                    analysis[f'has_{format_key}'] = True
                    analysis['formats'].add(format_name)
                    analysis['model_files'].append(file)
                    categorized = True
                    break
            
            if not categorized:
                if file_lower in config_file_names:
                    analysis['config_files'].append(file)
                else:
                    analysis['other_files'].append(file)
        
        analysis['formats'] = list(analysis['formats'])
        return analysis

    def _get_license_info(self, repo_id: str) -> str:
        """Extract license information from model card."""
        try:
            readme_path = hf_hub_download(repo_id=repo_id, filename="README.md", token=self.token)
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Look for license in YAML frontmatter
            lines = content.split('\n')
            for line in lines:
                if line.strip().lower().startswith('license:'):
                    return line.split(':', 1)[1].strip()
            
            return "Unknown"
        except Exception:
            return "Unknown"

    def _format_parameter_count(self, params: int) -> str:
        """Format parameter count in human readable form."""
        if params >= 1_000_000_000:
            return f"{params / 1_000_000_000:.1f}B"
        elif params >= 1_000_000:
            return f"{params / 1_000_000:.1f}M"
        else:
            return f"{params:,}"

    def _build_report_sections(self, repo_id: str, config: Dict, tokenizer: Dict, 
                              model_info: Dict, file_analysis: Dict, license_info: str) -> List[str]:
        """Build comprehensive markdown report sections."""
        sections = []
        
        # Header section
        sections.extend([
            f"# 🤗 Model Inspector Report",
            f"**Repository:** `{repo_id}`",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            ""
        ])
        
        # Repository stats section
        if model_info:
            stats_data = [
                ("Downloads", f"{model_info.get('downloads', 0):,}"),
                ("Likes", f"{model_info.get('likes', 0):,}"),
                ("Author", model_info.get('author', 'N/A')),
                ("Private", 'Yes' if model_info.get('private') else 'No'),
                ("Gated", 'Yes' if model_info.get('gated') else 'No'),
                ("Library", model_info.get('library_name', 'N/A')),
                ("Pipeline Tag", model_info.get('pipeline_tag', 'N/A')),
                ("License", license_info)
            ]
            
            sections.extend(["## 📊 Repository Stats"] + 
                          [f"- **{label}:** {value}" for label, value in stats_data] + 
                          [""])
            
            # Add dates if available
            date_fields = [
                ('created_at', 'Created'),
                ('last_modified', 'Last Modified')
            ]
            
            for field, label in date_fields:
                if model_info.get(field):
                    sections.append(f"- **{label}:** {model_info[field].strftime('%Y-%m-%d')}")
            
            sections.append("")
            
            # Tags section
            if model_info.get('tags'):
                tags_str = ", ".join([f"`{tag}`" for tag in model_info['tags'][:10]])
                sections.extend(["## 🏷️ Tags", tags_str, ""])
        
        # Architecture section
        if config:
            sections.append("## 🏗️ Model Architecture")
            
            # Basic architecture info
            basic_fields = [
                ('model_type', 'Model Type', lambda x: f"`{x}`"),
                ('hidden_size', 'Hidden Size', lambda x: f"{x:,}" if isinstance(x, int) else str(x)),
                ('num_hidden_layers', 'Number of Layers', str),
                ('num_attention_heads', 'Attention Heads', str),
                ('vocab_size', 'Vocabulary Size', lambda x: f"{x:,}" if isinstance(x, int) else str(x))
            ]
            
            for field, label, formatter in basic_fields:
                if field in config and config[field] is not None:
                    sections.append(f"- **{label}:** {formatter(config[field])}")
                else:
                    sections.append(f"- **{label}:** N/A")
            
            sections.append("")
            
            # Additional architecture details
            additional_fields = [
                ('intermediate_size', 'Intermediate Size'),
                ('max_position_embeddings', 'Max Position Embeddings'), 
                ('num_key_value_heads', 'Key-Value Heads'),
                ('rope_theta', 'RoPE Theta'),
                ('sliding_window', 'Sliding Window'),
                ('attention_dropout', 'Attention Dropout'),
                ('hidden_dropout', 'Hidden Dropout'),
                ('rope_scaling', 'RoPE Scaling')
            ]
            
            extra_details = []
            for field, label in additional_fields:
                if field in config:
                    value = config[field]
                    formatted_value = f"{value:,}" if isinstance(value, int) and value > 1000 else str(value)
                    extra_details.append(f"- **{label}:** {formatted_value}")
            
            if extra_details:
                sections.extend(extra_details + [""])
            
            # Parameter estimation
            params = self._estimate_parameters(config)
            if params:
                param_str = self._format_parameter_count(params)
                sections.extend([
                    f"- **Estimated Parameters:** ~{param_str} ({params:,})",
                    ""
                ])
            
            # Quantization detection
            is_quantized, detected_types = self._detect_quantization(config)
            if is_quantized:
                quant_types_str = ', '.join(sorted(detected_types))
                sections.append(f"- **Quantization:** ✅ Detected ({quant_types_str})")
            else:
                sections.append(f"- **Quantization:** ❌ Not detected")
            
            sections.append("")
        else:
            sections.extend(["## ⚠️ Model Architecture", "Could not load `config.json`", ""])
        
        # Tokenizer section
        if tokenizer:
            sections.extend([
                "## 📝 Tokenizer",
                f"- **Model Max Length:** {tokenizer.get('model_max_length', 'N/A'):,}" if isinstance(tokenizer.get('model_max_length'), int) else f"- **Model Max Length:** {tokenizer.get('model_max_length', 'N/A')}",
                f"- **Tokenizer Vocab Size:** {len(tokenizer.get('model', {}).get('vocab', {})) if tokenizer.get('model', {}).get('vocab') else 'N/A'}",
                ""
            ])
        
        # File analysis section
        format_indicators = [
            ('SafeTensors', 'has_safetensors'),
            ('PyTorch', 'has_pytorch'),
            ('GGUF', 'has_gguf'),
            ('ONNX', 'has_onnx')
        ]
        
        sections.extend([
            "## 📁 Repository Files",
            f"- **Total Files:** {file_analysis['total_files']}",
            f"- **Model File Formats:** {', '.join(file_analysis['formats']) if file_analysis['formats'] else 'None detected'}"
        ])
        
        sections.extend([
            f"- **{name}:** {'✅' if file_analysis.get(key) else '❌'}"
            for name, key in format_indicators
        ])
        
        sections.append("")
        
        # Model files listing
        if file_analysis['model_files']:
            sections.extend(["### Model Files", "```"])
            display_files = sorted(file_analysis['model_files'])[:20]
            sections.extend(display_files)
            
            if len(file_analysis['model_files']) > 20:
                sections.append(f"... and {len(file_analysis['model_files']) - 20} more")
            
            sections.extend(["```", ""])
        
        return sections

    def inspect_model(self, repo_id: str) -> str:
        """Main inspection method that orchestrates the analysis."""
        logger.info(f"🔍 Inspecting model: {repo_id}")
        
        # Verify repository exists
        if not repo_exists(repo_id, token=self.token):
            raise ValueError(f"Repository {repo_id} not found or not accessible")
        
        # Gather information
        logger.info("📥 Fetching model information...")
        model_info = self._safe_api_request(repo_id)
        
        logger.info("📥 Loading configuration files...")
        config = self._load_json_file(repo_id, "config.json")
        tokenizer = self._load_json_file(repo_id, "tokenizer.json")
        
        logger.info("📥 Analyzing repository files...")
        siblings = model_info.get('siblings', []) if model_info else []
        file_analysis = self._analyze_files(siblings)
        
        logger.info("📥 Fetching license information...")
        license_info = self._get_license_info(repo_id)
        
        # Generate report
        logger.info("📝 Generating comprehensive report...")
        sections = self._build_report_sections(
            repo_id, config, tokenizer, model_info, file_analysis, license_info
        )
        
        return "\n".join(sections)

    def save_report(self, report: str, output_path: Path = Path("model_inspection_report.md")) -> Dict[str, Any]:
        """Save report and return GitHub Actions outputs."""
        output_path.write_text(report, encoding="utf-8")
        
        # Extract outputs for GitHub Actions
        outputs = {}
        lines = report.split('\n')
        
        for line in lines:
            if '**Downloads:**' in line:
                try:
                    outputs['downloads'] = int(line.split('**Downloads:**')[1].strip().replace(',', ''))
                except (ValueError, IndexError):
                    outputs['downloads'] = 0
            elif '**Likes:**' in line:
                try:
                    outputs['likes'] = int(line.split('**Likes:**')[1].strip().replace(',', ''))
                except (ValueError, IndexError):
                    outputs['likes'] = 0
            elif '**Gated:**' in line:
                outputs['gated'] = 'Yes' in line
        
        return outputs, output_path


def main():
    """Main execution function for GitHub Actions compatibility."""
    # Load environment variables
    load_dotenv()

    # Detect execution context
    repo_id = os.getenv("INPUT_REPO_ID")
    hf_token = os.getenv("INPUT_HF_TOKEN") or os.getenv("HF_TOKEN")

    if not repo_id:
        parser = argparse.ArgumentParser(description="Enhanced Hugging Face Model Inspector")
        parser.add_argument("--repo-id", required=True, help="Hugging Face repo ID (e.g., microsoft/DialoGPT-large)")
        parser.add_argument("--hf-token", help="Hugging Face access token (for gated/private repos)")
        args = parser.parse_args()
        repo_id = args.repo_id
        hf_token = hf_token or args.hf_token

    try:
        # Initialize inspector
        inspector = ModelInspector(token=hf_token)
        
        # Generate report
        report = inspector.inspect_model(repo_id)
        
        # Display report
        print("\n" + "="*80)
        print(report)
        print("="*80)
        
        # Save report and get outputs
        outputs, output_path = inspector.save_report(report)
        
        # Single success print statement
        print(f"\n✅ Comprehensive report saved to {output_path.resolve()}")

        # Set GitHub Actions outputs if running in GA
        if os.getenv("GITHUB_ACTIONS"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write(f"report-path={output_path}\n")
                for key, value in outputs.items():
                    f.write(f"{key}={value}\n")

    except Exception as e:
        logger.error(f"Error during inspection: {e}")
        exit(1)


# Entry point for GitHub Actions
main()