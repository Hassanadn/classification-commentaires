import re
from bs4 import BeautifulSoup

class TextCleaner:
    def __init__(self, pipeline_config):
        self.pipeline = {step['name']: step['enabled'] for step in pipeline_config}
        self.exceptions = self._get_exceptions(pipeline_config)
    
    def _get_exceptions(self, pipeline_config):
        for step in pipeline_config:
            if step['name'] == 'remove_punctuation' and step['enabled']:
                return ''.join(step.get('exceptions', []))
        return ''
    
    def clean(self, text):
        if self.pipeline.get('remove_html', False):
            text = BeautifulSoup(text, 'html.parser').get_text()
        
        if self.pipeline.get('remove_urls', False):
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        if self.pipeline.get('remove_punctuation', False):
            pattern = f"[^\\w\\s{self.exceptions}]"
            text = re.sub(pattern, '', text)
        
        if self.pipeline.get('remove_numbers', False):
            text = re.sub(r'\d+', '', text)
        
        if self.pipeline.get('remove_extra_whitespaces', False):
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text