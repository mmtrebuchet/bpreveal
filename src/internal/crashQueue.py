"""A multiprocessing queue class that clears out the queue if an exception is raised."""
import multiprocessing as mp
from typing import Any
import queue
from bpreveal.internal.constants import QUEUE_TIMEOUT


class CrashQueue:
    """A queue that closes itself if a timeout expires during a put.

    :param maxsize: The maximum size of the queue. Default is unlimited.
    :param timeout: The time, in seconds, that blocking operations should
        wait before erroring out.

    This class fixes a problem where a producer is feeding objects into a queue and
    then the consumer crashes. In this instance, the producer thread will raise
    a queue.Full exception, but the process can't close because there's still
    data to write to the queue. But since the consumer has crashed, this creates
    deadlock. If a write to a queue in this class times out, it breaks that deadlock
    by allowing the process to exit before the queue is flushed.
    This class is, more or less, a subclass of multiprocessing.Queue.
    """

    q: mp.Queue
    timeout: float

    def __init__(self, maxsize: int = -1,
                 timeout: float = QUEUE_TIMEOUT):
        if maxsize > 0:
            self.q = mp.Queue(maxsize)
        else:
            self.q = mp.Queue()
        self.timeout = timeout

    def empty(self) -> bool:
        """Is the queue currently empty?"""
        return self.q.empty()

    def full(self) -> bool:
        """Is the queue currently full?"""
        return self.q.full()

    def put(self, obj: Any, timeout: float | None = None) -> None:
        """Put the given object on the queue. If timeout expires, clear the queue.

        :param obj: Any (picklable) object.
        :param timeout: The maximum time to wait, in seconds. If this expires,
            then the queue gets cancel_join_thread called on it and the queue.Full
            exception is raised.
        """
        if timeout is None:
            timeout = self.timeout
        try:
            self.q.put(obj, timeout=timeout)
        except queue.Full:
            self.q.cancel_join_thread()
            raise

    def get(self, timeout: float | None = None) -> Any:
        """Get an object from the queue.

        :param timeout: The number of seconds to wait before raising an exception.
            If omitted, the timeout given in the initializer.
        :return: The object from the queue.
        """
        if timeout is None:
            timeout = self.timeout
        return self.q.get(timeout=timeout)

    def close(self) -> None:
        """Indicate that no more items will be added to the queue.

        Automatically called when the queue is garbage collected.
        """
        self.q.close()
