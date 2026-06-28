import unittest
from unittest.mock import Mock

from src.app.progress import NullProgressSink, ProgressEvent, RichProgressSink


class TestNullProgressSink(unittest.TestCase):
    def test_is_a_noop_context_and_reuses_itself_for_children(self):
        sink = NullProgressSink()

        with sink as entered:
            entered.start("stage", "starting", total=2)
            entered.update("stage", "working", completed=1)
            entered.advance("stage")
            entered.finish("stage", "done")

        self.assertIs(entered, sink)
        self.assertIs(sink.child("nested"), sink)
        self.assertIsNone(sink.raw_progress)

    def test_group_exposes_key_bound_progress_operations(self):
        sink = NullProgressSink()

        with sink.group("stage", "Starting", total=2) as group:
            group.update("Working")
            group.advance()
            with group.group("nested", "Nested", total=1) as nested:
                nested.finish("Nested done")
            group.finish("Done")


class TestRichProgressSink(unittest.TestCase):
    def setUp(self):
        self.progress = Mock()
        self.progress.add_task.return_value = 17
        self.sink = RichProgressSink(self.progress)

    def test_start_update_advance_and_finish_target_the_same_task(self):
        self.sink.start("stage", "starting", total=4)
        self.sink.update("stage", "working", completed=2)
        self.sink.advance("stage", step=2, message="advanced")
        self.sink.finish("stage", "finished")

        self.progress.add_task.assert_called_once_with("starting", total=4)
        self.progress.advance.assert_called_once_with(17, 2)
        self.assertEqual(self.progress.update.call_args_list[-1].args, (17,))
        self.assertEqual(self.progress.update.call_args_list[-1].kwargs["description"], "finished")

    def test_child_prefixes_keys_and_indents_messages(self):
        surface = self.sink.child("protein").child("surface")
        pocket = self.sink.child("protein").child("pocket")
        self.progress.add_task.side_effect = [17, 18]

        surface.start("run", "Extracting", total=1)
        pocket.start("run", "Finding", total=1)

        self.assertEqual(self.progress.add_task.call_count, 2)
        self.assertEqual(self.progress.add_task.call_args_list[0].args, ("      Extracting",))
        self.assertEqual(self.progress.add_task.call_args_list[1].args, ("      Finding",))

    def test_advance_before_start_creates_a_task(self):
        self.sink.advance("stage", message="working")

        self.progress.add_task.assert_called_once_with("working", total=None)
        self.progress.advance.assert_called_once_with(17, 1)

    def test_finish_unknown_stage_with_no_message_is_noop(self):
        self.sink.finish("unknown")

        self.progress.add_task.assert_not_called()
        self.progress.update.assert_not_called()

    def test_rejects_unknown_event_action(self):
        event = ProgressEvent("unsupported", "stage")

        with self.assertRaisesRegex(ValueError, "Unsupported progress action"):
            self.sink.emit(event)

    def test_group_binds_messages_and_advances_to_its_stage_key(self):
        with self.sink.group("stage", "Starting", total=2) as group:
            group.update("Working")
            group.advance()
            group.finish("Done")

        self.progress.add_task.assert_called_once_with("Starting", total=2)
        self.progress.advance.assert_called_once_with(17, 1)
        self.assertEqual(self.progress.update.call_args_list[-1].kwargs["description"], "Done")


if __name__ == "__main__":
    unittest.main()
