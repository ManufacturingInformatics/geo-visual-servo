import pyrealsense2 as rs

class IntelFilters:
    
    def __init__(self):
        
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.filler = rs.hole_filling_filter()
    
    def filter(self, frame):
        
        frame = self.spatial.process(frame)
        frame = self.temporal.process(frame)
        frame = self.filler.process(frame)
        return frame