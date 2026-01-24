import streamlit.web.cli as stcli
import os
import sys
import argparse
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StreamlitAppRunner:
    def __init__(self, app_name: str = "Orbital Witness AI"):
        self.app_name = app_name
        self.project_root = Path(__file__).parent.resolve()
        self.app_path = self.project_root / "Interface" / "main.py"
        
    def validate_environment(self):
        if not self.app_path.exists():
            raise FileNotFoundError(
                f"Streamlit app not found at {self.app_path}"
            )
        
        required_dirs = ['Interface', 'app', 'knowledge_base']
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                logger.warning(f"Directory not found: {dir_path}")
        
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"App path: {self.app_path}")
        
    def setup_python_path(self):
        project_root_str = str(self.project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
            logger.info(f"Added to Python path: {project_root_str}")
        
    def parse_arguments(self):
        parser = argparse.ArgumentParser(
            description=f"Launch {self.app_name}",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        parser.add_argument(
            '--port',
            type=int,
            default=8501,
            help='Port number for the Streamlit server'
        )
        
        parser.add_argument(
            '--host',
            type=str,
            default='localhost',
            help='Host address for the Streamlit server'
        )
        
        parser.add_argument(
            '--server-headless',
            action='store_true',
            help='Run in headless mode without opening browser'
        )
        
        parser.add_argument(
            '--theme-base',
            type=str,
            choices=['light', 'dark'],
            default='dark',
            help='Base theme for the application'
        )
        
        parser.add_argument(
            '--browser-gather-usage-stats',
            action='store_true',
            help='Allow Streamlit to gather usage statistics'
        )
        
        parser.add_argument(
            '--log-level',
            type=str,
            choices=['debug', 'info', 'warning', 'error'],
            default='info',
            help='Logging level'
        )
        
        return parser.parse_args()
    
    def build_streamlit_args(self, args):
        streamlit_args = [
            "run",
            str(self.app_path),
            "--server.port", str(args.port),
            "--server.address", args.host,
            "--theme.base", args.theme_base,
            "--logger.level", args.log_level,
        ]
        
        if args.server_headless:
            streamlit_args.extend(["--server.headless", "true"])
        
        if not args.browser_gather_usage_stats:
            streamlit_args.extend(["--browser.gatherUsageStats", "false"])
        
        return streamlit_args
    
    def run(self):
        try:
            logger.info(f"Starting {self.app_name}...")
            
            self.validate_environment()
            self.setup_python_path()
            
            args = self.parse_arguments()
            streamlit_args = self.build_streamlit_args(args)
            
            logger.info(f"Launching Streamlit with args: {' '.join(streamlit_args)}")
            logger.info(f"Server will be available at http://{args.host}:{args.port}")
            
            stcli.main(streamlit_args)
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to start application: {e}")
            sys.exit(1)


def main():
    runner = StreamlitAppRunner(app_name="Orbital Witness AI")
    runner.run()


if __name__ == "__main__":
    main()