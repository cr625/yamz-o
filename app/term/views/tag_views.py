from app.term import term_blueprint as term
from app.term.forms import TagForm, AddTagForm
from app.term.models import Tag, Term
from flask import render_template, redirect, url_for, flash
from flask_login import login_required

@term.route("tag/create", methods=["GET", "POST"])
def create_tag():
    tag_form = TagForm()
    if tag_form.validate_on_submit():
        tag_category = tag_form.category.data
        tag_value = tag_form.value.data
        tag_description = tag_form.description.data
        tag_domain = tag_form.domain.data

        tag = Tag.query.filter_by(
            category=tag_category, value=tag_value).first()
        if (tag is None) or (tag_value.lower() != tag.value.lower()):
            new_tag = Tag(
                category=tag_category, value=tag_value, description=tag_description, domain=tag_domain
            )
            new_tag.save()
            return redirect(url_for("term.list_tags"))
        else:
            flash("Tag already exists")
            return redirect(url_for("term.edit_tag", tag_id=tag.id))

    else:
        return render_template("tag/create_tag.jinja", form=tag_form)


@term.route("tag/edit/<int:tag_id>", methods=["GET", "POST"])
@login_required
def edit_tag(tag_id):
    tag_form = TagForm()
    tag = Tag.query.get_or_404(tag_id)
    if tag_form.validate_on_submit():
        tag.category = tag_form.category.data
        tag.value = tag_form.value.data
        tag.description = tag_form.description.data
        tag.domain = tag_form.domain.data
        tag.save()
        flash(
            'Tag updated.  <small>[<a href="'
            + url_for("term.list_tags")
            + '">Return to list]</a></small>'
        )
        return redirect(url_for("term.edit_tag", tag_id=tag_id))
    tag_form.category.data = tag.category
    tag_form.value.data = tag.value
    tag_form.description.data = tag.description
    tag_form.domain.data = tag.domain
    return render_template("tag/edit_tag.jinja", form=tag_form)


@term.route("tag/delete/<int:tag_id>", methods=["GET", "POST"])
@login_required
def delete_tag(tag_id):
    tag = Tag.query.get_or_404(tag_id)
    tag.delete()
    return redirect(url_for("term.list_tags"))


@term.route("tag/add/<int:term_id>", methods=["GET", "POST"])
@login_required
def add_tag(term_id):
    tag_form = AddTagForm()
    term = Term.query.get_or_404(term_id)
    concept_id = term.concept_id
    tag = Tag.query.filter_by(value=tag_form.tag_list.data).first()
    term.tags.append(tag)
    term.save()
    return redirect(url_for("term.display_term", concept_id=concept_id))


@term.route("tag/remove/<int:term_id>/<int:tag_id>", methods=["GET", "POST"])
def remove_tag(term_id, tag_id):
    term = Term.query.get_or_404(term_id)
    tag = Tag.query.get_or_404(tag_id)
    term.tags.remove(tag)
    term.save()
    return redirect(url_for("term.display_term", concept_id=term.concept_id))


@term.route("tag/list")
def list_tags():
    tags = Tag.query.order_by(Tag.value)
    return render_template("tag/list_tags.jinja", tags=tags)

