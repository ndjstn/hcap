from io import BytesIO
import matplotlib.pyplot as plt
import json
import logging
import streamlit as st
import altair as alt

def fig_to_png(fig):
    """Convert a matplotlib or plotly figure to PNG bytes."""
    try:
        if isinstance(fig, plt.Figure):
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
            buf.seek(0)
            return buf.getvalue()
        else:
            # For plotly figures
            return fig.to_image(format='png')
    except Exception as e:
        logging.error(f"Error converting figure to PNG: {e}")
        return None

def chart_to_png(chart):
    """Convert an Altair chart to PNG bytes."""
    try:
        # Create a BytesIO buffer
        buf = BytesIO()
        # Save the chart as SVG to the buffer
        chart.save(buf, format='svg')
        buf.seek(0)
        
        # Convert SVG to PNG using cairosvg
        import cairosvg
        png_data = cairosvg.svg2png(bytestring=buf.getvalue())
        return png_data
        
    except ImportError:
        logging.warning("cairosvg not installed, falling back to JSON format")
        try:
            # Fallback to JSON if cairosvg is not available
            return json.dumps(chart.to_dict()).encode('utf-8')
        except Exception as e:
            logging.error(f"Error converting chart to JSON: {e}")
            return None
    except Exception as e:
        logging.error(f"Error converting chart to PNG: {e}")
        return None

def export_to_json(data):
    """Export data to JSON format."""
    try:
        return json.dumps(data, default=str, indent=2)
    except Exception as e:
        logging.error(f"Error exporting to JSON: {e}")
        return None

def setup_keyboard_shortcuts():
    """Setup keyboard shortcuts using JavaScript"""
    st.markdown("""
        <script>
            document.addEventListener('DOMContentLoaded', (event) => {
                // Auto-focus room number
                document.querySelector('input[aria-label="Room number"]').focus();
                
                // Keyboard shortcuts
                document.addEventListener('keydown', (e) => {
                    if (e.altKey) {
                        switch(e.key) {
                            case '1':
                                document.querySelector('[data-testid="all_always"]').click();
                                break;
                            case 's':
                                document.querySelector('[data-testid="stay_feedback"]').focus();
                                break;
                            case 'r':
                                document.querySelector('[data-testid="recognition"]').focus();
                                break;
                            case 'n':
                                location.reload();
                                break;
                        }
                    }
                });
            });
        </script>
    """, unsafe_allow_html=True)

def get_staff_suggestions(input_text):
    """Return staff name suggestions based on input"""
    # Mock staff list - replace with actual staff database
    staff_list = [
        "Nurse Smith",
        "Nurse Johnson",
        "Dr. Brown",
        "Nurse Williams",
        "Dr. Davis"
    ]
    
    query = input_text.split('@')[-1].lower()
    return [name for name in staff_list if query in name.lower()] 