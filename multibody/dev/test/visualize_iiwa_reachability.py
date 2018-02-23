from director import lcmUtils
from director import roboturdf
import pickle
import bot_core as lcmbotcore

def receiveMessage(msg):
    drake_path = '/home/hongkai/drake-distro-bk'

    #robotModel, jointController = roboturdf.loadRobotModel(urdfFile=drake_path+"/manipulation/models/iiwa_description/urdf/iiwa14_polytope_collision.urdf", view=view, useConfigFile=False)
    #jointController.setPose('my posture', np.zeros(len(jointController.jointNames)))

    folderName = 'my data'

    # remove the folder completely
    om.removeFromObjectModel(om.findObjectByName(folderName))

    #create a folder
    folder = om.getOrCreateContainer(folderName)

    # unpack message
    data = pickle.loads(msg.data)


    file = open(drake_path + '/iiwa_reachability_global_ik.txt', 'r')

    lines = file.readlines()

    line_number = 0

    num_orient = 15 
    d_reachable = []
    for i in range(num_orient + 1):
        d_reachable.append(DebugData())
    orient_count = 0
    num_reachable_orient = 0
    while line_number < len(lines):
        line = lines[line_number]
        if line.startswith("pos count:"):
            pos_count_str = lines[line_number].split(':')
            pos_count = int(pos_count_str[1])
        elif line.startswith("orient count:"):
            orient_count_str = lines[line_number].split(':')
            orient_count = int(orient_count_str[1])
        elif line.startswith("position:"):
            line_number = line_number + 1
            pos_str = lines[line_number].split()
            pos = [float(pos_str[0]), float(pos_str[1]), float(pos_str[2])]
        elif line.startswith("nonlinear ik info:"):
            nonlinear_ik_status_str = line.split(':')
            nonlinear_ik_status = int(nonlinear_ik_status_str[1])
            if (nonlinear_ik_status <= 10):
                num_reachable_orient = num_reachable_orient + 1
        elif line.startswith("global_ik info:"):
            global_ik_status_str = line.split(':')
            global_ik_status = int(global_ik_status_str[1])
        elif line.startswith("nonlinear ik resolve info:"):
            nonlinear_ik_resolve_status_str = line.split(':')
            nonlinear_ik_resolve_status = int(nonlinear_ik_status_str[1])

            if orient_count == 4:
                reachable_color = [float(num_orient - num_reachable_orient) / num_orient, float(num_reachable_orient) / num_orient, 0]
                d_reachable[num_reachable_orient].addSphere(pos, radius = 0.01, color = reachable_color)
                # reset num_reachable_orient
                num_reachable_orient = 0
        line_number = line_number + 1

    for i in range(num_orient + 1):
        name = 'reachable' + str(i)
        vis.showPolyData(d_reachable[i].getPolyData(), name, parent = folder, colorByName = 'RGB255')


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
