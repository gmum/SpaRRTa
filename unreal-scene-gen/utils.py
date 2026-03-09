import unreal 
import random
from typing import Tuple

from collections import deque
from collections.abc import Iterable


def detect_ground_at_position(x: float, y: float, base_z: float, search_range: float = 20.0) -> float:
    """
    Performs line trace to find ground height at X,Y position.
    Returns the Z coordinate of the ground surface.
    """
    # Start above expected ground level
    start_pos = unreal.Vector(x, y, base_z + search_range)
    # End below expected ground level
    end_pos = unreal.Vector(x, y, base_z - search_range)
    
    # Perform line trace to find ground
    hit_result = unreal.SystemLibrary.line_trace_single(
        world_context_object=unreal.EditorLevelLibrary.get_editor_world(),
        start=start_pos,
        end=end_pos,
        trace_channel=unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,  # World Static
        trace_complex=True,
        actors_to_ignore=[],
        draw_debug_type=unreal.DrawDebugTrace.NONE
    )
    
    hit_result = hit_result.to_tuple()
    
    if hit_result[0]:
        return hit_result[4].z
    else:
        # Fallback to base Z if no ground found
        print(f"No ground found at {x}, {y}, {base_z}")
        return base_z


class PyTick():
    _delegate_handle = None
    _current = None
    schedule = None

    def __init__(self):
        self.schedule = deque()
        self._delegate_handle = unreal.register_slate_post_tick_callback(self._callback)

    def _callback(self, _):
        if self._current is None:
            if self.schedule:
                self._current = self.schedule.popleft()
            else:
                print('🏁 All jobs done.')
                unreal.unregister_slate_post_tick_callback(self._delegate_handle)
                return

        try:
            task = next(self._current)
            if task is not None and isinstance(task, Iterable):
                self.schedule.appendleft(self._current)
                self._current = task

        except StopIteration:
            self._current = None
        except Exception as e:
            self._current = None
            print(f"⚠️ Error during task: {e}")
            raise


def destroy_by_tag(tag="SCRIPT_CREATED"):
    for actor in unreal.EditorLevelLibrary.get_all_level_actors():
        if any(str(t) == tag for t in actor.tags):
            unreal.EditorLevelLibrary.destroy_actor(actor)