import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ---------- constants ----------
REQUIRED_COLUMNS = ["Bay", "Row", "Tier", "Container_ID", "Container_Location", "Declared_Cargo", "Colour"]

# ---------- helpers ----------
def convert_row_to_y(row):
    """Converts ship row numbering to Cartesian Y-coordinates without any offsets."""
    try:
        r = int(row)
        if r == 0: return 0
        steps = (r + 1) // 2
        sign = 1 if r % 2 == 1 else -1
        return sign * steps
    except (ValueError, TypeError):
        return None

def get_bay_slot_index(bay):
    """
    Returns the slot index for a bay. This is used to calculate spacing.
    For 40ft bays (even): slot index = (bay - 2) / 2
    For 20ft bays (odd): they belong to a 40ft slot, so we find which one:
      - Odd bays at the forward end (17, 21, 25...): slot = (bay + 1 - 2) / 2
      - Odd bays at the aft end (19, 23, 27...): slot = (bay - 1 - 2) / 2
    """
    if bay % 2 == 0:  # 40ft bay
        return (bay - 2) // 2
    else:  # 20ft bay
        slot_number = (bay - 1) // 2
        if slot_number % 2 == 0:
            return (bay + 1 - 2) // 2
        else:
            return (bay - 1 - 2) // 2

def create_container_traces(x, y, z, length, width, height, color, hovertext):
    """Creates traces for a container's solid faces and wireframe edges."""
    vertices = {
        'x': [x, x + length, x + length, x, x, x + length, x + length, x],
        'y': [y, y, y + width, y + width, y, y, y + width, y + width],
        'z': [z, z, z, z, z + height, z + height, z + height, z + height]
    }

    face_mesh = go.Mesh3d(
        x=vertices['x'], y=vertices['y'], z=vertices['z'],
        i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3],
        j = [1, 3, 5, 7, 4, 7, 2, 6, 3, 7, 4, 5],
        k = [2, 2, 6, 6, 7, 3, 6, 5, 7, 6, 5, 1],
        color=color, opacity=1.0, hoverinfo="text", text=hovertext,
        flatshading=True, lighting=dict(ambient=0.8, diffuse=0.2, specular=0.0)
    )

    path_indices = [0,1,2,3,0, None, 4,5,6,7,4, None, 0,4, None, 1,5, None, 2,6, None, 3,7]
    edge_x, edge_y, edge_z = [], [], []
    for i in path_indices:
        if i is None:
            edge_x.append(None); edge_y.append(None); edge_z.append(None)
        else:
            edge_x.append(vertices['x'][i]); edge_y.append(vertices['y'][i]); edge_z.append(vertices['z'][i])

    edge_wireframe = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines', line=dict(color='black', width=2), hoverinfo='none'
    )
    
    return [face_mesh, edge_wireframe]


def create_stowage_plot(df, longitudinal_separation=0.0, lateral_offset=0.0):
    """Creates the 3D stowage plot from a dataframe."""
    
    # Validate columns
    if not all(col in df.columns for col in REQUIRED_COLUMNS):
        st.error(f"CSV must contain the following columns: {REQUIRED_COLUMNS}")
        return None

    # Clean data
    df = df.dropna(subset=["Bay", "Row", "Tier"]).copy()
    df = df.astype({"Bay": int, "Row": int, "Tier": int})

    fig = go.Figure()
    
    # Process each container
    for _, c in df.iterrows():
        try:
            bay = c["Bay"]
            z = c["Tier"] - 1

            # --- Calculate X Position (Longitudinal) ---
            if bay % 2 == 0:  # 40ft container
                base_x = (bay - 2) / 2
                length = 2.0
            else:  # 20ft container
                base_x = (bay - 1) / 2
                length = 1.0
            
            slot_idx = get_bay_slot_index(bay)
            x_final = base_x + (slot_idx * longitudinal_separation)
            
            # --- Calculate Y Position (Lateral) ---
            y_base = convert_row_to_y(c["Row"])
            if y_base is None:
                continue
            
            y_final = y_base + (slot_idx * lateral_offset)

            # --- Create hover text ---
            hover_text = (
                f"<b>ID:</b> {c['Container_ID']}<br>"
                f"<b>Location:</b> {c['Container_Location']}<br>"
                f"<b>Cargo:</b> {c['Declared_Cargo']}"
            )
            
            # --- Create container traces ---
            container_traces = create_container_traces(
                x=x_final, y=y_final, z=z,
                length=length, width=1.0, height=1.0,
                color=c["Colour"], hovertext=hover_text
            )
            for trace in container_traces:
                fig.add_trace(trace)

        except (ValueError, TypeError) as e:
            continue

    # --- Prepare Custom Axis Ticks ---
    unique_even_bays = sorted([b for b in df['Bay'].unique() if b % 2 == 0])
    bay_tickvals = []
    for b in unique_even_bays:
        base_x = (b - 2) / 2
        slot_idx = get_bay_slot_index(b)
        offset = slot_idx * longitudinal_separation
        bay_tickvals.append(base_x + offset)
        
    bay_ticktext = [f"{int(b):02d}" for b in unique_even_bays]

    unique_tiers = sorted(df['Tier'].unique())
    tier_tickvals = [t - 1 for t in unique_tiers]
    tier_ticktext = [str(t) for t in unique_tiers]

    unique_rows = sorted(df['Row'].unique())
    row_tickvals = [convert_row_to_y(r) for r in unique_rows]
    row_ticktext = [str(r) for r in unique_rows]

    fig.update_layout(
        title='Vessel Stowage Plan', showlegend=False,
        scene=dict(
            xaxis=dict(title="Bay", tickvals=bay_tickvals, ticktext=bay_ticktext),
            yaxis=dict(title="Row", tickvals=row_tickvals, ticktext=row_ticktext),
            zaxis=dict(title="Tier", tickvals=tier_tickvals, ticktext=tier_ticktext),
            aspectmode="data", camera=dict(eye=dict(x=1.8, y=-1.8, z=1.5))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=700
    )
    
    return fig


# ---------- Streamlit App ----------
def main():
    st.set_page_config(page_title="Container Stowage Visualization", layout="wide")
    
    st.title("🚢 Container Stowage Visualization")
    st.markdown("Upload your stowage data CSV to visualize the 3D container arrangement")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        st.divider()
        
        # Separation controls
        st.subheader("Bay Separation")
        longitudinal_sep = st.slider(
            "Longitudinal Separation",
            min_value=0.0,
            max_value=3.0,
            value=0.0,
            step=0.1,
            help="Increase spacing between bays along the ship's length"
        )
        
        lateral_offset = st.slider(
            "Lateral Offset",
            min_value=0.0,
            max_value=3.0,
            value=0.0,
            step=0.1,
            help="Increase spacing between bays across the ship's width"
        )
        
        st.divider()
        
        # Info
        st.subheader("Required CSV Columns")
        st.markdown("""
        - Bay
        - Row
        - Tier
        - Container_ID
        - Container_Location
        - Declared_Cargo
        - Colour
        """)
    
    # Main content area
    if uploaded_file is not None:
        try:
            # Read the CSV
            df = pd.read_csv(uploaded_file)
            
            # Display info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Containers", len(df))
            with col2:
                st.metric("Unique Bays", df['Bay'].nunique())
            with col3:
                st.metric("Max Tier", df['Tier'].max())
            
            # Create and display the plot
            fig = create_stowage_plot(df, longitudinal_sep, lateral_offset)
            
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                
                # Optional: Show data table
                with st.expander("View Data Table"):
                    st.dataframe(df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        # Show placeholder when no file is uploaded
        st.info("👈 Please upload a CSV file to begin")
        
        # Optional: Show example data format
        with st.expander("Example CSV Format"):
            example_data = pd.DataFrame({
                'Bay': [18, 18, 17, 19],
                'Row': [1, 3, 1, 1],
                'Tier': [82, 82, 82, 82],
                'Container_ID': ['CONT001', 'CONT002', 'CONT003', 'CONT004'],
                'Container_Location': ['180182', '180382', '170182', '190182'],
                'Declared_Cargo': ['Electronics', 'Clothing', 'Food', 'Furniture'],
                'Colour': ['red', 'blue', 'green', 'yellow']
            })
            st.dataframe(example_data)

if __name__ == "__main__":
    main()
