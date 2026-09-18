import unittest
from unittest.mock import Mock,patch
import queue_worker


class WorkerBrandingFailureTests(unittest.TestCase):
    def test_branding_failure_does_not_escape_worker(self):
        database=Mock();database.execute.return_value.rowcount=0
        with patch.object(queue_worker,'connect',return_value=database),patch.object(queue_worker,'setting',return_value='2500'),patch.object(queue_worker,'load_one_minute',return_value=0),patch.object(queue_worker,'available_ram_mb',return_value=4000),patch.object(queue_worker,'claim_next',return_value={'id':'test-job'}),patch.object(queue_worker,'process_job',side_effect=RuntimeError('unsafe source logo')),patch('sys.argv',['queue_worker','--once']),patch('builtins.print'):
            self.assertEqual(queue_worker.main(),0)
        database.rollback.assert_called_once()
        updates=[call for call in database.execute.call_args_list if "status='FAILED'" in call.args[0]]
        self.assertEqual(len(updates),1)
        self.assertEqual(updates[0].args[1][0],'unsafe source logo')
