from collections import namedtuple
import numpy as np


GameState = namedtuple(
    "GameState",
    ["count", "timestamp", "ball_pos", "robot_poses"]
)

class VelApprox:
    def __init__(self, expectedvels, gs: GameState, playerDecay: float, ballDecay: float):
        """
        Initializes
        weights for calculating new vels: float[0, 1]
        ball velocity decay value: float
        player velocity decay value: float
        
        previous gamestate: GameState
        
        calculated ball vel for reporting: numpy 1d float arrays
        
        expected velocities: Dict
        robot team A vels: list[numpy 1d float arrays]
        robot team B vels: list[numpy 1d float arrays]
        """
        self.VELWEIGHT = 0.5
        self.BDECAY = ballDecay
        self.PDECAY = playerDecay
        
        self.prevGS = gs
        
        self.ballVel = np.zeros(2)
        
        self.expVel = expectedvels
        self.robotAVels = [np.zeros(2) for _ in gs.robot_poses["TritonBots"]]
        self.robotBVels = [np.zeros(2) for _ in gs.robot_poses["TeamB"]]
    
    def dictToPosVect(self, robotDict):
        (_, (x, y, _theta)) = next(iter(robotDict.items()))
        return np.array([x, y], dtype=float)
    
    
    def teamVelCalculation(self, robotvels, gs: GameState, teamName: str, dt: float):
        oldRobotPos = self.prevGS.robot_poses[teamName]
        newrobotpos = gs.robot_poses[teamName]
        for i in range(len(newrobotpos)):
            newpos = self.dictToPosVect(newrobotpos[i])
            oldpos = self.dictToPosVect(oldRobotPos[i])
            
            dsRobot = newpos-oldpos
            newRobotVel = dsRobot/dt
            
            robotvels[i] = self.VELWEIGHT*self.expVel.expectedVels[teamName]+self.VELWEIGHT*newRobotVel
        return robotvels
        
    
    def update(self, gs: GameState):
        """
        Function that updates the calculated ball vels, ball acc, robot vels based on
        newly inputed GameState Data
        """
        # calcualte time and distance differential of ball
        dt = gs.timestamp-self.prevGS.timestamp
        dsBall = np.array(gs.ball_pos)-np.array(self.prevGS.ball_pos)
        
        # calculate instantaneous ball velocity and speed
        newBallVel = dsBall/dt
        newBallSpeed = np.linalg.norm(newBallVel)
        
        # for calculating ball acc check if ball has been kicked (if speed has increased, if vel vector points in direction greater than 90 deg)
        if newBallSpeed > np.linalg.norm(self.ballVel)*(1.2) or np.dot(newBallVel, self.ballVel) <= 0 or self.ballVel == np.zeros(2):
            self.ballVel = newBallVel
        else:
            self.ballVel = self.VELWEIGHT*self.ballVel*self.BDECAY+(1-self.VELWEIGHT)*newBallVel
            
        self.robotAVels = self.teamVelCalculation(self.robotAVels, gs, "TritonBots", dt)
        self.robotBVels = self.teamVelCalculation(self.robotBVels, gs, "TeamB", dt)
        
        self.prevGS = gs
        
    def getBallState(self):
        return self.ballVel
    
    
    def getRobotState(self):
        return self.robotAVels, self.robotBVels
    
    
    def __str__(self):
        """
        Returns string of the stored calculated ball vel, acc, and robot vels
        """
        return f"Calculated Ball Velocity: {self.ballVel} \nCalculated Robot Team A Velocities: {self.robotAVels} \n\nCalculated Robot Team B Velocities: {self.robotBVels}"
    
    
