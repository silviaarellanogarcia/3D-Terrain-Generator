from werkzeug.wsgi import FileWrapper

from flask import Flask, render_template, request, Response
from pipeline import heightmap_generation, super_resolution, define_texture, apply_texture, render_blender, \
    render_result

app = Flask(__name__)


@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")


@app.route("/generate_heightmap", methods=['GET'])
def generate_heightmap():
    # call the heightmap_generation function
    hmap_b64 = heightmap_generation()
    # pass the heightmap image to the template
    return render_template('index.html', hmap_b64=hmap_b64)


@app.route("/save_heightmap", methods=['POST'])
def save_heightmap():
    hmap_b64 = request.form['hmap_b64']
    sr_b64 = super_resolution(hmap_b64)
    return render_template('index.html', sr_b64=sr_b64)


@app.route('/define_texture', methods=['POST'])
def define_textures():
    # Once I have all my textures defined, I access here
    texture = request.form.getlist('texture')
    lower_threshold = request.form.getlist('lower_threshold')
    upper_threshold = request.form.getlist('upper_threshold')
    position = request.form.getlist('position')

    ranges = []
    for i in range(len(texture)):
        ranges.extend(define_texture(texture[i], lower_threshold[i], upper_threshold[i], position[i]))

    # Here I have the 'ranges' variable complete. I can start making the textures
    sr_b64 = request.form['sr_b64']
    texture_b64, result_hmaps, result_seps = apply_texture(ranges, sr_b64)

    return render_template('index.html', sr_b64=sr_b64, texture_b64=texture_b64, result_hmaps=result_hmaps, result_seps=result_seps)


@app.route('/save_render', methods=['POST'])
def save_render():
    sr_b64 = request.form['sr_b64']
    texture_b64 = request.form['texture_b64']
    hmap_seps = request.form.getlist('result_hmaps')
    texture_seps = request.form.getlist('result_seps')
    action = request.form.get('action', 'download')
    if action == 'download':
        zip_buffer = render_blender(sr_b64, texture_b64)

        for i in range(len(hmap_seps)):
           zip_buffer = render_blender(hmap_seps[i], texture_seps[i], buffer=zip_buffer, name=f'Separate_{i}')

        zip_buffer.seek(0) # Useful to read the file from the beginning. If we didn't do it, the pointer would be at the last position where we wrote.
        zip_file = FileWrapper(zip_buffer)
        headers = {'Content-Disposition': 'attachment; filename="MyTerrain.zip"'}
        return Response(zip_file, direct_passthrough=True, mimetype='application/zip', headers=headers)
    file_bytes = render_result(sr_b64, texture_b64)
    gltf_file = FileWrapper(file_bytes)
    headers = {'Content-Disposition': 'attachment; filename="MyTerrain.gltf"'}
    return Response(gltf_file, direct_passthrough=True, mimetype='model/gltf-binary', headers=headers)

