from director import lcmUtils
from director import roboturdf
import pickle
import bot_core as lcmbotcore

def receiveMessage(msg):
    drake_path = '/home/hongkai/drake-distro'

    #robotModel, jointController = roboturdf.loadRobotModel(urdfFile=drake_path+"/drake/manipulation/models/iiwa_description/urdf/iiwa14_polytope_collision.urdf", view=view, useConfigFile=False)
    #jointController.setPose('my posture', np.zeros(len(jointController.jointNames)))

    #ee_pose = np.array([0, 0, 0, 0, 0, 0, 0])
    #ee_joint_controller.setPose('ee posture', ee_pose)
    folderName = 'my data'

    # remove the folder completely
    om.removeFromObjectModel(om.findObjectByName(folderName))

    #create a folder
    folder = om.getOrCreateContainer(folderName)

    # unpack message
    data = pickle.loads(msg.data)

    d_reachable = DebugData()
    d_unreachable = DebugData()
    d_unknown = DebugData()

    file = open(drake_path + '/iiwa_reachability_global_ik.txt', 'r')

    lines = file.readlines()

    line_number = 0

    while line_number < len(lines):
        line = lines[line_number]
        if line.startswith("position:"):
            line_number = line_number + 1
            pos_str = lines[line_number].split()
            pos = [float(pos_str[0]), float(pos_str[1]), float(pos_str[2])]
        elif line.startswith("nonlinear ik info:"):
            nonlinear_ik_status_str = line.split(':')
            nonlinear_ik_status = int(nonlinear_ik_status_str[1])
        elif line.startswith("global_ik info:"):
            global_ik_status_str = line.split(':')
            global_ik_status = int(global_ik_status_str[1])
        elif line.startswith("nonlinear ik resolve info:"):
            nonlinear_ik_resolve_status_str = line.split(':')
            nonlinear_ik_resolve_status = int(nonlinear_ik_status_str[1])

            if nonlinear_ik_resolve_status <= 10 or nonlinear_ik_resolve_status <= 10 :
                d_reachable.addSphere(pos, radius = 0.01, color = [0, 1, 0])
            elif global_ik_status == -2:
                d_unreachable.addSphere(pos, radius = 0.01, color = [1, 0, 0])
            else:
                d_unknown.addSphere(pos, radius = 0.01, color = [0, 0, 1])
        line_number = line_number + 1

    vis.showPolyData(d_reachable.getPolyData(), 'reachable', parent = folder, colorByName = 'RGB255')
    vis.showPolyData(d_unreachable.getPolyData(), 'unreachable', parent = folder, colorByName = 'RGB255')
    vis.showPolyData(d_unknown.getPolyData(), 'unknown', parent = folder, colorByName = 'RGB255')


def publishData():
    data = 1
    msg = lcmbotcore.raw_t()
    msg.data = pickle.dumps(data)
    msg.length = len(msg.data)
    lcmUtils.publish('MY_DATA', msg)


# add an lcm subscriber with a python callback
lcmUtils.addSubscriber('MY_DATA', messageClass=lcmbotcore.raw_t, callback=receiveMessage)

# publish an lcm message
publishData()

# call myTimer.start() to begin publishing
myTimer = TimerCallback(callback = publishData, targetFps = 10)
