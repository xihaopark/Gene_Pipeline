#!/usr/bin/env python3
"""
GenBank Gene Extractor - 服务器启动脚本
用于在服务器环境中自动启动应用，无需手动执行streamlit命令
支持直接运行: python run.py
"""

import os
import sys
import subprocess
import time
import signal
import threading
import socket
import webbrowser
from pathlib import Path
from typing import Optional, Tuple

class StreamlitLauncher:
    """Streamlit应用启动器"""
    
    def __init__(self):
        self.process = None
        self.current_dir = Path(__file__).parent.absolute()
        self.main_file = self.current_dir / "main.py"
        
    def check_dependencies(self) -> bool:
        """检查所有必要的依赖是否已安装"""
        required_packages = [
            'streamlit', 'biopython', 'pandas', 'requests', 
            'anthropic', 'scikit-learn', 'numpy', 'plotly'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ 缺少以下依赖包: {', '.join(missing_packages)}")
            return False
        
        print("✅ 所有依赖包检查通过")
        return True
    
    def install_requirements(self) -> bool:
        """安装依赖包"""
        requirements_file = self.current_dir / "requirements.txt"
        if not requirements_file.exists():
            print("❌ 未找到requirements.txt文件")
            return False
        
        print("📦 正在安装依赖包...")
        try:
            # 使用pip安装依赖
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ 依赖包安装完成")
                return True
            else:
                print(f"❌ 依赖包安装失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ 依赖包安装超时")
            return False
        except Exception as e:
            print(f"❌ 依赖包安装失败: {e}")
            return False
    
    def find_available_port(self, start_port: int = 8501) -> int:
        """找到可用的端口"""
        for port in range(start_port, start_port + 100):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(('0.0.0.0', port))
                    return port
            except OSError:
                continue
        return start_port
    
    def setup_environment(self):
        """设置环境变量和工作目录"""
        # 设置工作目录
        os.chdir(self.current_dir)
        
        # 添加当前目录到Python路径
        if str(self.current_dir) not in sys.path:
            sys.path.insert(0, str(self.current_dir))
        
        # 设置Streamlit环境变量
        streamlit_config = {
            'STREAMLIT_SERVER_HEADLESS': 'true',
            'STREAMLIT_SERVER_ENABLE_CORS': 'false',
            'STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION': 'false',
            'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false',
            'STREAMLIT_GLOBAL_DEVELOPMENT_MODE': 'false'
        }
        
        for key, value in streamlit_config.items():
            os.environ[key] = value
        
        print("✅ 环境配置完成")
    
    def validate_main_file(self) -> bool:
        """验证main.py文件是否存在且有效"""
        if not self.main_file.exists():
            print(f"❌ 未找到main.py文件: {self.main_file}")
            return False
        
        # 简单检查main.py是否包含streamlit代码
        try:
            with open(self.main_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'streamlit' not in content.lower():
                    print("⚠️  警告: main.py文件可能不是有效的Streamlit应用")
                    return False
        except Exception as e:
            print(f"❌ 读取main.py文件失败: {e}")
            return False
        
        print("✅ main.py文件验证通过")
        return True
    
    def start_streamlit(self, port: Optional[int] = None, host: str = '0.0.0.0', 
                       open_browser: bool = False) -> bool:
        """启动Streamlit应用"""
        
        if port is None:
            port = self.find_available_port()
        
        # 构建streamlit命令
        cmd = [
            sys.executable, "-m", "streamlit", "run", str(self.main_file),
            "--server.port", str(port),
            "--server.address", host,
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXSRFProtection", "false",
            "--browser.gatherUsageStats", "false"
        ]
        
        # 显示启动信息
        print("\n" + "="*60)
        print("🚀 GenBank Gene Extractor 正在启动...")
        print("="*60)
        print(f"📍 服务器地址: http://{host}:{port}")
        if host == '0.0.0.0':
            print(f"🌐 本地访问: http://localhost:{port}")
            print(f"🌐 网络访问: http://[服务器IP]:{port}")
        print("⏹️  按 Ctrl+C 停止应用")
        print("="*60)
        
        try:
            # 启动streamlit进程
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 等待应用启动
            startup_success = self._wait_for_startup(port, timeout=30)
            
            if startup_success:
                print("✅ 应用启动成功!")
                
                # 可选择打开浏览器
                if open_browser:
                    try:
                        webbrowser.open(f"http://localhost:{port}")
                    except:
                        pass
                
                # 实时输出日志
                self._stream_output()
                
            else:
                print("❌ 应用启动失败")
                return False
                
        except KeyboardInterrupt:
            print("\n⏹️  正在停止应用...")
            self._stop_process()
            print("✅ 应用已停止")
            return True
            
        except Exception as e:
            print(f"❌ 启动失败: {e}")
            return False
        
        return True
    
    def _wait_for_startup(self, port: int, timeout: int = 30) -> bool:
        """等待应用启动完成"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    result = s.connect_ex(('localhost', port))
                    if result == 0:
                        return True
            except:
                pass
            
            # 检查进程是否还在运行
            if self.process and self.process.poll() is not None:
                return False
                
            time.sleep(0.5)
            print(".", end="", flush=True)
        
        return False
    
    def _stream_output(self):
        """实时输出应用日志"""
        def output_reader():
            try:
                if self.process and self.process.stdout:
                    for line in iter(self.process.stdout.readline, ''):
                        if line.strip():
                            # 过滤掉一些不必要的日志
                            if not any(skip in line.lower() for skip in ['you can now view', 'local url', 'network url']):
                                print(f"📋 {line.rstrip()}")
            except:
                pass
        
        output_thread = threading.Thread(target=output_reader)
        output_thread.daemon = True
        output_thread.start()
        
        # 等待进程结束
        if self.process:
            self.process.wait()
    
    def _stop_process(self):
        """停止streamlit进程"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
    
    def run(self, port: Optional[int] = None, host: str = '0.0.0.0', 
            open_browser: bool = False, auto_install: bool = True) -> bool:
        """运行应用的主方法"""
        
        print("🔧 正在初始化...")
        
        # 1. 设置环境
        self.setup_environment()
        
        # 2. 验证main.py文件
        if not self.validate_main_file():
            return False
        
        # 3. 检查依赖
        if not self.check_dependencies():
            if auto_install:
                print("🔄 尝试自动安装依赖...")
                if not self.install_requirements():
                    print("❌ 请手动安装依赖: pip install -r requirements.txt")
                    return False
                # 重新检查依赖
                if not self.check_dependencies():
                    print("❌ 依赖安装后仍然缺少某些包")
                    return False
            else:
                print("❌ 请先安装依赖: pip install -r requirements.txt")
                return False
        
        # 4. 启动应用
        return self.start_streamlit(port, host, open_browser)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='GenBank Gene Extractor 服务器启动脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run.py                    # 默认启动 (端口8501)
  python run.py --port 8080        # 指定端口
  python run.py --host 127.0.0.1   # 仅本地访问
  python run.py --browser           # 启动后打开浏览器
  python run.py --no-install       # 不自动安装依赖
        """
    )
    
    parser.add_argument('--port', type=int, default=None, 
                       help='指定端口号 (默认: 8501)')
    parser.add_argument('--host', type=str, default='0.0.0.0', 
                       help='指定主机地址 (默认: 0.0.0.0, 允许外部访问)')
    parser.add_argument('--browser', action='store_true', 
                       help='启动后自动打开浏览器')
    parser.add_argument('--no-install', action='store_true', 
                       help='不自动安装缺少的依赖包')
    
    args = parser.parse_args()
    
    # 创建启动器并运行
    launcher = StreamlitLauncher()
    
    try:
        success = launcher.run(
            port=args.port,
            host=args.host,
            open_browser=args.browser,
            auto_install=not args.no_install
        )
        
        if not success:
            print("\n❌ 应用启动失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 再见!")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 意外错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 