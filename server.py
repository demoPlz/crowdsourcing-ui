import os
from flask import Flask, jsonify
from flask_cors import CORS  # 1. Import the CORS library
from urdf_parser_py.urdf import URDF

app = Flask(__name__)
CORS(app)  # 2. Initialize CORS with your app. This enables it for all routes.

STATIC_FOLDER = 'static'
URDF_PATH = os.path.join(STATIC_FOLDER, 'wxai_base.urdf')

@app.route('/get_initial_joint_positions')
def get_initial_joint_positions():
    try:
        robot = URDF.from_xml_file(URDF_PATH)
        joint_positions = {}
        for joint in robot.joints:
            if joint.joint_type in ['revolute', 'prismatic']:
                joint_positions[joint.name] = 0
        return jsonify(joint_positions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)