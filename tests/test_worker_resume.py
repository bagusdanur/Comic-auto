import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
from PIL import Image
from queue_worker import process_job


class WorkerResumeTests(unittest.TestCase):
    def test_completed_page_keeps_output_and_does_not_run_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);jobroot=root/'job'
            original=jobroot/'original'/'001.png';original.parent.mkdir(parents=True)
            output=jobroot/'output'/'001.png';output.parent.mkdir()
            page=jobroot/'metadata'/'page-001';page.mkdir(parents=True)
            for f in (original,output,page/'typeset.png'):Image.new('RGB',(20,20),'white').save(f)
            for name in ('typeset-report.json','translations.json','ocr-result.json'):(page/name).write_text('{}')
            (page/'pipeline-status.json').write_text(json.dumps({'status':'COMPLETED','input':str(original),'review_ids':[]}))
            database=Mock();database.execute.return_value.fetchone.return_value={'status':'COMPLETED','output_path':str(output)}
            job={'id':'test','input_path':str(original),'output_path':str(jobroot/'chapter.zip'),'options':{'batch':True,'input_files':['001.png']}}
            before=output.read_bytes()
            with patch('queue_worker.subprocess.Popen') as run,patch('queue_worker.archive_webp') as archive,patch('builtins.print'):
                process_job(database,job,root,resume=True)
            run.assert_not_called();archive.assert_called_once()
            self.assertEqual(before,output.read_bytes())
            self.assertFalse(any('DELETE FROM images' in call.args[0] for call in database.execute.call_args_list))
