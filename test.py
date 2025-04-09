import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class DisplacementPublisher(Node):
  def __init__(self):
      super().__init__('displacement_publisher')
      self.publisher2 = self.create_publisher(Float32MultiArray, '/finger', 10)
      self.timer_finger = self.create_timer(1/120, self.publish_finger)

  def publish_finger(self, left_finger):
      msg = Float32MultiArray()
      msg.data = left_finger.tolist()
      self.publisher2.publish(msg)
      self.get_logger().info(f'Published left finger data: {left_finger}')

class Sim:
    def __init__(self, print_freq=True):
        rclpy.init()
        self.publisher = DisplacementPublisher()
        self.print_freq = print_freq
        self.target_pose = np.array([-0.4, 0.08, 1.1]) 
        self.threshold = 0.05
        self.active_position = None
        self.active_orientation = None
        self.status = "Inactive"
        self.displacement = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.left_finger = nps.zeros(12)
        self.previous_pose = np.array([-0.35, 0.08, 1.1])
        self.publisher.publish_finger(self.left_finger)    

if __name__ == '__main__':
    Sim()
