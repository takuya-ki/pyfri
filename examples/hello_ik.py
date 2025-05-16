import argparse
import sys

# FRI Client: https://github.com/cmower/FRI-Client-SDK_Python
import pyfri as fri

# NumPy: https://numpy.org/
import numpy as np

np.set_printoptions(precision=5, suppress=True, linewidth=1000)

# Local scripts
from ik import IK



class HelloIKClient(fri.LBRClient):
    def __init__(self, ik, action_sequence):
        super().__init__()
        self.ik = ik
        self.actions = action_sequence
        self.torques = np.zeros(fri.LBRState.NUMBER_OF_JOINTS)
        self.done = False

    def monitor(self):
        pass

    def onStateChange(self, old_state, new_state):
        print(f"State changed from {old_state} to {new_state}")

        if new_state == fri.ESessionState.MONITORING_READY:
            self.q = None
            self.torques = np.zeros(fri.LBRState.NUMBER_OF_JOINTS)

    def waitForCommand(self):
        self.q = self.robotState().getIpoJointPosition()
        self.robotCommand().setJointPosition(self.q.astype(np.float32))

        if self.robotState().getClientCommandMode() == fri.EClientCommandMode.TORQUE:
            self.robotCommand().setTorque(self.torques.astype(np.float32))

        if self.done:
            print("Waiting to exit session gracefully.")

    def command(self):
        if self.done:
            current_pos = self.robotState().getMeasuredJointPosition()
            self.robotCommand().setJointPosition(current_pos.astype(np.float32))
            return

        try:
            vg = self.actions.pop(0)
        except IndexError:
            print("All actions completed.")
            self.done = True
            return

        self.q = self.ik(self.q, vg, self.robotState().getSampleTime())

        self.robotCommand().setJointPosition(self.q.astype(np.float32))
        if self.robotState().getClientCommandMode() == fri.EClientCommandMode.TORQUE:
            self.robotCommand().setTorque(self.torques.astype(np.float32))


def args_factory():
    parser = argparse.ArgumentParser(description="LRBJointSineOverlay example.")
    parser.add_argument(
        "--hostname",
        dest="hostname",
        default=None,
        help="The hostname used to communicate with the KUKA Sunrise Controller.",
    )
    parser.add_argument(
        "--port",
        dest="port",
        type=int,
        default=30200,
        help="The port number used to communicate with the KUKA Sunrise Controller.",
    )
    parser.add_argument(
        "--lbr-ver",
        dest="lbr_ver",
        type=int,
        choices=[7, 14],
        required=True,
        help="The KUKA LBR Med version number.",
    )

    return parser.parse_args()


def main():
    print("Running FRI Version:", fri.FRI_CLIENT_VERSION)

    args = args_factory()
    ik = IK(args.lbr_ver)

    _action_sequence = []
    z_values = [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04, -0.04]
    interval = 13

    for z_val in z_values:
        _action_sequence.append([0., 0., z_val, 0., 0., 0.])
        _action_sequence.extend([[0., 0., 0., 0., 0., 0.]] * interval)

    _action_sequence.extend([[0., 0., 0., 0., 0., 0.]] * interval)

    client = HelloIKClient(ik, _action_sequence)
    app = fri.ClientApplication(client)
    success = app.connect(args.port, args.hostname)

    if not success:
        print("Connection to KUKA Sunrise controller failed.")
        return 1

    print("Connection to KUKA Sunrise controller established.")

    try:
        while success:
            success = app.step()

            if client.robotState().getSessionState() == fri.ESessionState.IDLE:
                break

    except KeyboardInterrupt:
        pass

    except SystemExit:
        pass

    finally:
        app.disconnect()
        print("Goodbye")

    return 0


if __name__ == "__main__":
    sys.exit(main())
